"""Diffusion-style classifier: one F*N window that diffuses between the image and the one-hot.

Not classical (random-noise) diffusion. The state is a single window of WIN = f*N bits, which is
both the input and the output of the net:

  - the CLEAN target (noise level 0) is the one-hot rep: slot j (class j%C) is 1 iff image class
    == j%C;
  - the fully-NOISED state (level 1) is the image embedding tiled across the window (what we
    actually have at inference);
  - at level alpha each bit is, independently, the image bit (prob alpha) or the target bit
    (else). So noise moves the target toward the input.

We build + CD one boolean-gate net (same machinery as grow_lut) to map a noisy window -> the clean
one-hot, training over a spread of noise levels. At inference we start the window at the image
(level 1) and apply the SAME net many times -- each pass denoises a bit, its output is the next
pass's input -- then read argmax class popcount.

    .venv/bin/python scratch/grow_diffuse.py --device cuda --train-size 0 --window-factor 8 --iters 16
"""

from __future__ import annotations

import argparse
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
        self.WIN = win
        self.H = win // n_classes
        self.max_gates = max_gates
        self.device = device
        self.pw = (1 << torch.arange(fan_in, device=device)).long()
        self.class_of = torch.arange(win, device=device) % n_classes
        self.tile = torch.arange(win, device=device) % n_inputs    # slot -> image input bit
        self.def_in = torch.full((win, fan_in), -1, dtype=torch.long, device=device)
        self.def_tt = torch.zeros((win, self.TT), dtype=torch.bool, device=device)
        self.depth = torch.zeros(win, dtype=torch.long, device=device)
        self.usage = torch.zeros(win, dtype=torch.long, device=device)
        self.buildable = torch.arange(win, device=device)
        self.n_gates_built = 0

    def set_inputs(self, input_bits):
        self.D = input_bits.shape[1]
        self.Wwords = (self.D + 63) // 64
        self.img = pack_bits(input_bits.to(self.device))[self.tile].contiguous()   # (WIN, W) tiled

    def _onehot(self, slots, y, bidx):
        return (self.class_of[slots][:, None] == y[bidx][None, :]).to(torch.float32)

    def _noisy(self, slots, bidx, y, alpha):
        """Window values at a random per-sample noise level <= alpha: each bit is the image bit
        (prob a) or the one-hot target bit (else). a=0 -> clean one-hot, a=1 -> image."""
        img = gather_batch(self.img, slots, bidx)              # image bits (m, Bb)
        oh = self._onehot(slots, y, bidx)
        a = torch.rand(bidx.shape[0], device=self.device) * alpha
        use_img = torch.rand((slots.shape[0], bidx.shape[0]), device=self.device) < a[None, :]
        return torch.where(use_img, img, oh)

    # -- build: correlate noisy-window inputs with the clean one-hot target ---------------
    def build_sweep(self, y, bidx, n_build, noise=1.0, max_feats=16384, chunk=8192):
        if self.n_gates_built >= self.max_gates:
            return 0
        n = min(n_build, self.WIN, self.max_gates - self.n_gates_built)
        tgt = self.buildable[torch.randperm(self.WIN, device=self.device)[:n]]
        feats = self.buildable
        if feats.numel() > max_feats:
            feats = feats[torch.randperm(feats.numel(), device=self.device)[:max_feats]]
        fb = self._noisy(feats, bidx, y, noise)                # noisy window inputs
        fb = fb - fb.mean(1, keepdim=True)
        r = self._onehot(tgt, y, bidx)                         # clean target
        rc = r - r.mean(1, keepdim=True)
        best_val = torch.zeros((n, self.K), device=self.device)
        best_fi = torch.zeros((n, self.K), dtype=torch.long, device=self.device)
        for s in range(0, n, chunk):
            cov = fb @ rc[s:s + chunk].T
            cov = cov.masked_fill(feats[:, None] == tgt[s:s + chunk][None, :], 0.0)
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
        self.def_in[slots] = ins
        self.def_tt[slots] = tt
        self.depth[slots] = self.depth[ins].max(1).values + 1
        self.usage.index_add_(0, ins.reshape(-1), torch.ones(ins.numel(), dtype=torch.long,
                                                             device=self.device))
        self.n_gates_built += slots.numel()

    # -- CD: flip a random bit, keep if the gate matches the clean target on more samples -
    def cd_pass(self, y, bidx, n_flip, noise=1.0):
        gates = (self.def_in[:, 0] >= 0).nonzero(as_tuple=False).flatten()
        if gates.numel() == 0:
            return 0
        cand = gates[torch.randperm(gates.numel(), device=self.device)[:min(n_flip, gates.numel())]]
        g = cand.shape[0]
        gr = torch.arange(g, device=self.device)
        kbit = torch.randint(self.TT, (g,), device=self.device)
        ins = self.def_in[cand]
        bits = self._noisy(ins.reshape(-1), bidx, y, noise).long().view(g, self.K, -1)
        cell = (bits * self.pw.view(1, self.K, 1)).sum(1)
        cur_tt = self.def_tt[cand]
        new_tt = cur_tt.clone(); new_tt[gr, kbit] = ~new_tt[gr, kbit]
        cur = cur_tt.gather(1, cell).to(torch.float32)
        new = new_tt.gather(1, cell).to(torch.float32)
        tb = self._onehot(cand, y, bidx)
        keep = ((new == tb).sum(1) - (cur == tb).sum(1)) > 0
        if not keep.any():
            return 0
        self.def_tt[cand[keep]] = new_tt[keep]
        return int(keep.sum().item())

    @torch.no_grad()
    def evaluate(self, input_bits, y, iters, batch=8192):
        """Start the window at the image (level 1), apply the denoiser `iters` times (output ->
        next input), read argmax class popcount."""
        d = input_bits.shape[1]
        built = (self.def_in[:, 0] >= 0).nonzero(as_tuple=False).flatten()
        ins_b, tt_b = self.def_in[built], self.def_tt[built]
        correct = 0
        for i in range(0, d, batch):
            xb = input_bits[:, i:i + batch].to(self.device)
            ww = (xb.shape[1] + 63) // 64
            state = pack_bits(xb)[self.tile].contiguous()      # window = image (noise level 1)
            for _ in range(max(1, iters)):
                state[built] = apply_gate(state[ins_b], tt_b)  # denoise step (output -> next input)
            score = torch.zeros((self.C, xb.shape[1]), device=self.device)
            for s0 in range(0, self.WIN, 8192):
                sl = slice(s0, min(s0 + 8192, self.WIN))
                b = unpack_bits(state[sl], xb.shape[1]).to(torch.float32)
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
    p.add_argument("--max-gates", type=int, default=150000)
    p.add_argument("--build-per-phase", type=int, default=2000)
    p.add_argument("--cd-per-phase", type=int, default=60000)
    p.add_argument("--cd-flips", type=int, default=4096)
    p.add_argument("--build-batch", type=int, default=2048)
    p.add_argument("--cd-batch", type=int, default=8192)
    p.add_argument("--noise", type=float, default=1.0, help="max rep corruption toward image")
    p.add_argument("--iters", type=int, default=16, help="diffusion steps at inference")
    p.add_argument("--eval-every", type=int, default=10)
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
    print(f"N={n} WIN={circ.WIN} H={circ.H} K={circ.K}", flush=True)

    def rb(k):
        return torch.randint(circ.D, (min(k, circ.D),), device=dev)

    t0 = time.time()
    print(f"\n  ph |  gates  | va_i1 | va_i{args.iters} | te_i{args.iters} | time", flush=True)
    best = -1.0
    ph = 0
    while circ.n_gates_built < args.max_gates:
        ph += 1
        circ.build_sweep(ytr, rb(args.build_batch), args.build_per_phase, noise=args.noise)
        done = 0
        while done < args.cd_per_phase:
            nf = min(args.cd_flips, args.cd_per_phase - done)
            circ.cd_pass(ytr, rb(args.cd_batch), nf, noise=args.noise); done += nf
        if ph % args.eval_every == 0:
            v1 = circ.evaluate(Xva, vy, iters=1)
            vk = circ.evaluate(Xva, vy, iters=args.iters)
            best = max(best, v1, vk)
            print(f"{ph:4d} | {circ.n_gates_built:7d} | {v1:5.2f} | {vk:5.2f} | "
                  f"{circ.evaluate(Xte, ey, iters=args.iters):5.2f} | {time.time() - t0:4.0f}s",
                  flush=True)
    print(f"\nFINAL best_val={best:.2f}", flush=True)


if __name__ == "__main__":
    main()
