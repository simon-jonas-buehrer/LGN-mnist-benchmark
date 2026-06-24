"""Dense one-hot 'diffusion' classifier grown by greedy correlation + coordinate descent.

Instead of the margin/hinge objective of grow_lut, this denoises the output representation toward
its one-hot target, image-conditioned:

  - a read-only IMAGE BANK of N thermometer bits (the condition),
  - a REP window of H*C = f*N slots that is the output representation. Slot j belongs to class
    j % C and its TARGET is the one-hot bit (1 if the image's class == j%C, else 0).

Each rep slot is therefore an independent class-c detector with a clear 0/1 target, so supervision
is dense and per-slot exact (no cross-class margin coupling). Build: correlate the pool
(bank + already-built rep slots) with each slot's residual (target - current) and wire the top-K
into a LUT. CD: flip a random truth-table bit, keep it if the slot matches its one-hot target on
more of the batch. Prediction = argmax over classes of the popcount of that class's rep slots.

(This is the one-shot dense-target version; iterative/recurrent diffusion is a later step.)

    .venv/bin/python scratch/grow_diffuse.py --device cuda --train-size 0 --window-factor 8
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Thermometer  # noqa: E402
from train import load_cifar10  # noqa: E402
from grow_lut import pack_bits, unpack_bits, gather_batch, apply_gate  # noqa: E402


class DiffuseCircuit:
    def __init__(self, n_inputs, n_classes, window_factor, *, fan_in, max_gates, device):
        self.N = n_inputs
        self.C = n_classes
        self.K = fan_in
        self.TT = 1 << fan_in
        win = int(round(window_factor * n_inputs))
        win -= win % n_classes
        self.WIN = win                                   # rep slots = H*C
        self.H = win // n_classes
        self.max_gates = max_gates
        self.device = device
        self.pw = (1 << torch.arange(fan_in, device=device)).long()

        # signal table: 0..N-1 = image bank (read-only), N..N+WIN-1 = rep slots (buildable head)
        self.bank0 = win                                 # rep occupies 0..WIN-1, bank appended after
        self.S = win + n_inputs
        self.class_of = torch.arange(win, device=device) % n_classes
        self.depth = torch.zeros(win, dtype=torch.long, device=device)
        self.usage = torch.zeros(self.S, dtype=torch.long, device=device)
        self.def_in = torch.full((win, fan_in), -1, dtype=torch.long, device=device)  # -1 = unbuilt
        self.def_tt = torch.zeros((win, self.TT), dtype=torch.bool, device=device)
        self.buildable = torch.arange(win, device=device)
        self.bank_idx = torch.arange(win, self.S, device=device)   # bank signal indices
        self.ops = []
        self.n_gates_built = 0

    def set_inputs(self, input_bits):
        d = input_bits.shape[1]
        self.D = d
        self.Wwords = (d + 63) // 64
        self.sig = torch.zeros((self.S, self.Wwords), dtype=torch.int64, device=self.device)
        self.sig[self.bank0:] = pack_bits(input_bits.to(self.device))   # bank; rep starts at 0

    def _target(self, y, bidx, slots):
        """One-hot target bit for the given rep slots on the batch: (len(slots), Bb)."""
        return (y[bidx][None, :] == self.class_of[slots][:, None]).to(torch.float32)

    # -- build: correlate pool with (target - current) per rep slot ----------------------
    def build_sweep(self, y, bidx, n_build, max_feats=16384, chunk=8192):
        if self.n_gates_built >= self.max_gates:
            return 0
        n = min(n_build, self.WIN, self.max_gates - self.n_gates_built)
        tgt = self.buildable[torch.randperm(self.WIN, device=self.device)[:n]]
        # pool = bank + already-built rep slots
        built = (self.def_in[:, 0] >= 0).nonzero(as_tuple=False).flatten()
        feats = torch.cat([self.bank_idx, built])
        if feats.numel() > max_feats:
            feats = feats[torch.randperm(feats.numel(), device=self.device)[:max_feats]]
        fb = gather_batch(self.sig, feats, bidx)
        fb = fb - fb.mean(1, keepdim=True)
        v = gather_batch(self.sig, tgt, bidx)                  # current rep value
        r = self._target(y, bidx, tgt) - v                    # dense residual in {-1,0,1}
        rc = r - r.mean(1, keepdim=True)

        best_val = torch.zeros((n, self.K), device=self.device)
        best_fi = torch.zeros((n, self.K), dtype=torch.long, device=self.device)
        for s in range(0, n, chunk):
            cov = fb @ rc[s:s + chunk].T
            self_mask = feats[:, None] == tgt[s:s + chunk][None, :]
            cov = cov.masked_fill(self_mask, 0.0)
            val, idx = cov.abs().topk(min(self.K, cov.shape[0]), dim=0)
            best_fi[s:s + chunk] = idx.T
            best_val[s:s + chunk] = cov.gather(0, idx).T
        ins = feats[best_fi]
        pol = best_val >= 0
        cell = (pol.long() * self.pw).sum(1)
        tt = torch.zeros((n, self.TT), dtype=torch.bool, device=self.device)
        tt[torch.arange(n, device=self.device), cell] = True
        self._write(tgt, ins, tt)
        return n

    def _write(self, slots, ins, tt):
        out = apply_gate(self.sig[ins], tt)
        self.sig[slots] = out
        self.def_in[slots] = ins
        self.def_tt[slots] = tt
        self.depth[slots] = torch.where(ins < self.WIN, self.depth[ins.clamp(max=self.WIN - 1)],
                                        torch.zeros_like(ins)).max(1).values + 1
        self.usage.index_add_(0, ins.reshape(-1), torch.ones(ins.numel(), dtype=torch.long,
                                                             device=self.device))
        self.ops.append((slots.clone(), ins.clone(), tt.clone()))
        self.n_gates_built += slots.numel()

    # -- CD: flip a random truth-table bit, keep if the slot matches its target more ------
    def cd_pass(self, y, bidx, n_flip):
        gates = (self.def_in[:, 0] >= 0).nonzero(as_tuple=False).flatten()
        if gates.numel() == 0:
            return 0
        cand = gates[torch.randperm(gates.numel(), device=self.device)[:min(n_flip, gates.numel())]]
        g = cand.shape[0]
        gr = torch.arange(g, device=self.device)
        kbit = torch.randint(self.TT, (g,), device=self.device)
        ins = self.def_in[cand]                                # (G, K)
        bits = gather_batch(self.sig, ins.reshape(-1), bidx).long().view(g, self.K, -1)
        cell = (bits * self.pw.view(1, self.K, 1)).sum(1)      # (G, Bb)
        cur_tt = self.def_tt[cand]
        new_tt = cur_tt.clone(); new_tt[gr, kbit] = ~new_tt[gr, kbit]
        cur = cur_tt.gather(1, cell).to(torch.float32)
        new = new_tt.gather(1, cell).to(torch.float32)
        tb = self._target(y, bidx, cand)                       # (G, Bb) one-hot target
        # keep flip iff it matches the target on strictly more batch samples (per-slot, exact)
        gain = ((new == tb).sum(1) - (cur == tb).sum(1))
        keep = gain > 0
        if not keep.any():
            return 0
        sel = cand[keep]
        sel_tt = new_tt[keep]
        self.sig[sel] = apply_gate(self.sig[self.def_in[sel]], sel_tt)
        self.def_tt[sel] = sel_tt
        self.ops.append((sel.clone(), self.def_in[sel].clone(), sel_tt.clone()))
        return int(keep.sum().item())

    @torch.no_grad()
    def evaluate(self, input_bits, y, batch=8192, iters=0):
        """iters=0: one feedforward pass (replay op order). iters>0: use the net as a diffusion
        step -- apply all rep gates synchronously `iters` times, denoising the rep each step."""
        d = input_bits.shape[1]
        built = (self.def_in[:, 0] >= 0).nonzero(as_tuple=False).flatten()
        correct = 0
        for i in range(0, d, batch):
            xb = input_bits[:, i:i + batch].to(self.device)
            ww = (xb.shape[1] + 63) // 64
            sig = torch.zeros((self.S, ww), dtype=torch.int64, device=self.device)
            sig[self.bank0:] = pack_bits(xb)
            if iters <= 0:
                for slots, ins, tt in self.ops:                 # feedforward replay
                    sig[slots] = apply_gate(sig[ins], tt)
            else:
                ins_b, tt_b = self.def_in[built], self.def_tt[built]
                for _ in range(iters):                          # recurrent diffusion steps
                    sig[built] = apply_gate(sig[ins_b], tt_b)   # all rep gates from current state
            score = torch.zeros((self.C, xb.shape[1]), device=self.device)
            for s0 in range(0, self.WIN, 8192):
                sl = slice(s0, min(s0 + 8192, self.WIN))
                b = unpack_bits(sig[sl], xb.shape[1]).to(torch.float32)
                score.index_add_(0, self.class_of[sl], b)
            correct += (score.argmax(0).cpu() == y[i:i + batch]).sum().item()
        return 100.0 * correct / d


def encode(images, enc):
    return enc(images).flatten(1).t().contiguous().to(torch.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/cifar-10-batches-py"))
    p.add_argument("--download", action="store_true")
    p.add_argument("--train-size", type=int, default=0)
    p.add_argument("--num-bits", type=int, default=5)
    p.add_argument("--window-factor", type=float, default=8.0)
    p.add_argument("--fan-in", type=int, default=4)
    p.add_argument("--max-gates", type=int, default=200000)
    p.add_argument("--build-per-phase", type=int, default=2000)
    p.add_argument("--cd-per-phase", type=int, default=60000)
    p.add_argument("--cd-flips", type=int, default=4096)
    p.add_argument("--build-batch", type=int, default=2048)
    p.add_argument("--cd-batch", type=int, default=8192)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--iters", type=int, default=8, help="diffusion steps at inference (recurrent)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    dev = args.device
    print(f"device={dev} args={vars(args)}", flush=True)
    tx, ty, ex, ey = load_cifar10(args.data_dir, args.download)
    wx, wy = (tx, ty) if args.train_size == 0 else (tx[:args.train_size], ty[:args.train_size])
    nv = max(1, round(len(wx) * 0.1))
    vx, vy, px, py = wx[-nv:], wy[-nv:], wx[:-nv], wy[:-nv]
    print(f"train={len(px)} val={len(vx)} test={len(ex)}", flush=True)
    enc = Thermometer(num_bits=args.num_bits).fit(px[:2000])
    Xtr, Xva, Xte = encode(px, enc), encode(vx, enc), encode(ex, enc)
    n = Xtr.shape[0]
    ytr = py.to(dev)
    circ = DiffuseCircuit(n, 10, args.window_factor, fan_in=args.fan_in,
                          max_gates=args.max_gates, device=dev)
    circ.set_inputs(Xtr)
    print(f"N={n} WIN(rep)={circ.WIN} H={circ.H} K={circ.K}", flush=True)

    def rb(k):
        return torch.randint(circ.D, (min(k, circ.D),), device=dev)

    t0 = time.time()
    print(f"\n  ph |  gates  | tr_ffn | va_ffn | va_dif({args.iters}) | te_dif | time", flush=True)
    best = -1.0
    ph = 0
    while circ.n_gates_built < args.max_gates:
        ph += 1
        circ.build_sweep(ytr, rb(args.build_batch), args.build_per_phase)
        done = 0
        while done < args.cd_per_phase:
            nf = min(args.cd_flips, args.cd_per_phase - done)
            circ.cd_pass(ytr, rb(args.cd_batch), nf); done += nf
        if ph % args.eval_every == 0:
            vaf = circ.evaluate(Xva, vy, iters=0)               # feedforward
            vad = circ.evaluate(Xva, vy, iters=args.iters)      # diffusion (recurrent)
            best = max(best, vaf, vad)
            print(f"{ph:4d} | {circ.n_gates_built:7d} | {circ.evaluate(Xtr, py, iters=0):6.2f} | "
                  f"{vaf:6.2f} | {vad:6.2f} | {circ.evaluate(Xte, ey, iters=args.iters):6.2f} | "
                  f"{time.time() - t0:4.0f}s", flush=True)
    print(f"\nFINAL best_val(ffn or diffuse)={best:.2f}", flush=True)


if __name__ == "__main__":
    main()
