"""Two-stage convolutional logic network grown by greedy correlation + coordinate descent.

The flat model (grow_lut.py) tops out ~41% val because it has no spatial structure. This adds a
convolutional front end:

  Phase 1 (conv): an image (B, Cb, H, W) of thermometer bits -> (B, M, H, W). Each of the M output
    channels is one boolean gate that reads a small local receptive field (K inputs given as
    offsets (channel-bit, dy, dx) within a (2R+1)^2 window) and is SHARED across every spatial
    position. The gates are built + CD'd as a classifier (GroupSum over the whole output map, a
    channel's class = m % C), so they learn translation-shared, class-useful local patterns.

  Phase 2 (dense): flatten the (frozen) conv output to N' = M*H*W bits and run the existing flat
    GrownCircuit on them exactly as before (window f*N', GroupSum head, build + CD).

Everything is binary and bitpacked along the image axis.

    .venv/bin/python scratch/grow_conv.py --device cuda --train-size 0 \
        --conv-channels 64 --radius 1 --conv-fan-in 4 --conv-gates 4000 \
        --window-factor 2 --dense-gates 200000
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
from grow_lut import GrownCircuit, pack_bits, unpack_bits  # noqa: E402

_SH = None


def _sh(dev):
    global _SH
    if _SH is None or _SH.device != dev:
        _SH = torch.arange(64, dtype=torch.int64, device=dev)
    return _SH


def popcount(words: torch.Tensor) -> torch.Tensor:
    """(..., W) int64 -> (...) number of set bits (SWAR)."""
    x = words
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    return ((x * 0x0101010101010101) >> 56).sum(-1)


# ======================================================================================
# Phase 1: a shared convolutional logic layer
# ======================================================================================
class ConvLayer:
    def __init__(self, cb: int, hw: int, n_classes: int, *, channels: int, radius: int,
                 fan_in: int, device: str):
        self.Cb, self.HW, self.C = cb, hw, n_classes
        self.M, self.R, self.K = channels, radius, fan_in
        self.dev = device
        self.TT = 1 << fan_in
        self.pw = (1 << torch.arange(fan_in, device=device)).long()
        offs = [(c, dy, dx) for c in range(cb)
                for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
        self.off = torch.tensor(offs, device=device)          # (n_off, 3): channel-bit, dy, dx
        self.n_off = len(offs)
        self.class_of = torch.arange(channels, device=device) % n_classes
        self.g_off = torch.zeros((channels, fan_in), dtype=torch.long, device=device)
        self.g_tt = torch.zeros((channels, self.TT), dtype=torch.bool, device=device)
        self.built = torch.zeros(channels, dtype=torch.bool, device=device)
        self.tau = math.sqrt(channels / n_classes * hw * hw)  # ~ class-summed count scale

    # -- packed shifted input maps ------------------------------------------------------
    def set_inputs(self, xmaps: torch.Tensor) -> None:
        """xmaps: (Cb, HW, HW, D) uint8 -> packed (Cb, HW, HW, W). Also caches all shifted maps
        for every offset so build/eval just index them."""
        self.D = xmaps.shape[-1]
        self.W = (self.D + 63) // 64
        Xp = pack_bits(xmaps.reshape(self.Cb * self.HW * self.HW, self.D)).reshape(
            self.Cb, self.HW, self.HW, self.W).to(self.dev)
        # precompute the shifted map for each candidate offset -> (n_off, HW, HW, W)
        self.shift = torch.zeros((self.n_off, self.HW, self.HW, self.W), dtype=torch.int64,
                                 device=self.dev)
        hw = self.HW
        for o in range(self.n_off):
            c, dy, dx = self.off[o].tolist()
            ys0, ys1 = max(0, -dy), min(hw, hw - dy)
            xs0, xs1 = max(0, -dx), min(hw, hw - dx)
            self.shift[o, ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx] = Xp[c, ys0:ys1, xs0:xs1]
        self.omap = torch.zeros((self.M, self.HW, self.HW, self.W), dtype=torch.int64, device=self.dev)
        self.count = torch.zeros((self.M, self.D), dtype=torch.float32, device=self.dev)  # firings/img
        self.score = torch.zeros((self.C, self.D), dtype=torch.float32, device=self.dev)

    def _apply(self, off_idx: torch.Tensor, tt: torch.Tensor) -> torch.Tensor:
        """Packed output map (HW,HW,W) of one gate from its K offset indices + truth table."""
        ins = self.shift[off_idx]                              # (K, HW, HW, W)
        out = torch.zeros_like(ins[0])
        for cell in range(self.TT):
            if not tt[cell]:
                continue
            m = torch.full_like(ins[0], -1)
            for i in range(self.K):
                m = m & (ins[i] if (cell >> i) & 1 else ~ins[i])
            out = out | m
        return out

    def _count(self, omap: torch.Tensor) -> torch.Tensor:
        """(HW,HW,W) -> (D,) firings per image (number of positions firing)."""
        bits = unpack_bits(omap.reshape(-1, self.W), self.D)   # (HW*HW, D) uint8
        return bits.sum(0).to(torch.float32)

    # -- supervision (multiclass hinge subgradient over the conv-count head) -------------
    def direction(self, y, bidx):
        logit = self.score[:, bidx] / self.tau
        ar = torch.arange(bidx.shape[0], device=self.dev)
        yb = y[bidx]
        sy = logit[yb, ar]
        d = torch.zeros_like(logit)
        d[logit > (sy - 1.0)] = -1.0
        o = logit.clone(); o[yb, ar] = -1e9
        d[yb, ar] = torch.where(sy >= o.max(0).values + 1.0, 0.0, 1.0)
        return d

    # -- build: pick K offsets by correlation with the channel's class residual ----------
    def build_sweep(self, y, bidx, n_build):
        d = self.direction(y, bidx)                            # (C, Bb)
        tgt = torch.randperm(self.M, device=self.dev)[:n_build]   # channels to (re)build
        bb = bidx.shape[0]
        # unpack each candidate offset's bit at every position for the batch images:
        word, offb = bidx >> 6, (bidx & 63).to(torch.int64)
        sb = ((self.shift[..., word] >> offb) & 1).to(torch.float32)   # (n_off, HW, HW, bb)
        ob = sb.reshape(self.n_off, -1)                        # (n_off, HW*HW*bb)
        ob = ob - ob.mean(1, keepdim=True)
        for m in tgt.tolist():
            c = m % self.C
            r = d[c][None, None, :].expand(self.HW, self.HW, bb).reshape(-1)   # residual per (y,x,img)
            r = r - r.mean()
            cov = ob @ r                                       # (n_off,)
            top = cov.abs().topk(self.K).indices
            pol = cov[top] >= 0
            tt = torch.zeros(self.TT, dtype=torch.bool, device=self.dev)
            tt[int((pol.long() * self.pw).sum())] = True       # AND seed at the polarity cell
            self._write(m, top, tt)
        return tgt.numel()

    def _write(self, m, off_idx, tt):
        new = self._apply(off_idx, tt)
        old_c = self.count[m].clone()
        self.omap[m] = new
        self.count[m] = self._count(new)
        self.score[self.class_of[m]] += self.count[m] - old_c
        self.g_off[m] = off_idx
        self.g_tt[m] = tt
        self.built[m] = True

    # -- CD: random truth-table bit flip on a random channel, keep if it lowers hinge ----
    def cd_pass(self, y, bidx, n_flip):
        ch = self.built.nonzero(as_tuple=False).flatten()
        if ch.numel() == 0:
            return 0
        ch = ch[torch.randperm(ch.numel(), device=self.dev)[:min(n_flip, ch.numel())]]
        ar = torch.arange(bidx.shape[0], device=self.dev)
        yb = y[bidx]
        kept = 0
        for m in ch.tolist():                                  # sequential (cross-position pooling)
            c = m % self.C
            bit = int(torch.randint(self.TT, (1,)))
            tt = self.g_tt[m].clone(); tt[bit] = ~tt[bit]
            new = self._apply(self.g_off[m], tt)
            new_c = self._count(new)
            base = self._hinge(self.score, y, bidx)
            sc = self.score.clone(); sc[c] += new_c - self.count[m]
            if self._hinge(sc, y, bidx) < base - 1e-6:
                self.score = sc; self.omap[m] = new; self.count[m] = new_c; self.g_tt[m] = tt
                kept += 1
        return kept

    def _hinge(self, score, y, bidx):
        L = score[:, bidx] / self.tau
        ar = torch.arange(bidx.shape[0], device=self.dev)
        yb = y[bidx]
        sy = L[yb, ar]
        o = L.clone(); o[yb, ar] = -1e9
        return torch.clamp(1.0 + o.max(0).values - sy, min=0).sum().item()

    # -- produce the conv output bit-maps for any images (for eval / phase 2) ------------
    @torch.no_grad()
    def transform(self, xmaps: torch.Tensor) -> torch.Tensor:
        """xmaps (Cb,HW,HW,D) uint8 -> conv output bits (M*HW*HW, D) uint8 (the dense input)."""
        d = xmaps.shape[-1]
        w = (d + 63) // 64
        Xp = pack_bits(xmaps.reshape(self.Cb * self.HW * self.HW, d)).reshape(
            self.Cb, self.HW, self.HW, w).to(self.dev)
        hw = self.HW
        shift = torch.zeros((self.n_off, hw, hw, w), dtype=torch.int64, device=self.dev)
        for o in range(self.n_off):
            c, dy, dx = self.off[o].tolist()
            ys0, ys1 = max(0, -dy), min(hw, hw - dy)
            xs0, xs1 = max(0, -dx), min(hw, hw - dx)
            shift[o, ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx] = Xp[c, ys0:ys1, xs0:xs1]
        out = torch.zeros((self.M, hw, hw, w), dtype=torch.int64, device=self.dev)
        for m in range(self.M):
            if not self.built[m]:
                continue
            ins = shift[self.g_off[m]]
            o_ = torch.zeros_like(ins[0])
            for cell in range(self.TT):
                if not self.g_tt[m, cell]:
                    continue
                mk = torch.full_like(ins[0], -1)
                for i in range(self.K):
                    mk = mk & (ins[i] if (cell >> i) & 1 else ~ins[i])
                o_ = o_ | mk
            out[m] = o_
        bits = unpack_bits(out.reshape(self.M * hw * hw, w), d)   # (M*HW*HW, D)
        return bits

    @torch.no_grad()
    def evaluate(self, xmaps, y) -> float:
        """Phase-1 conv-only top-1 accuracy via the count head."""
        bits = self.transform(xmaps).to(torch.float32)         # (M*HW*HW, D)
        bits = bits.reshape(self.M, self.HW * self.HW, -1).sum(1)   # (M, D) counts
        score = torch.zeros((self.C, xmaps.shape[-1]), device=self.dev)
        score.index_add_(0, self.class_of, bits)
        pred = score.argmax(0).cpu()
        return 100.0 * (pred == y).float().mean().item()


# ======================================================================================
# Encoding + driver
# ======================================================================================
def encode_maps(images: torch.Tensor, enc: Thermometer) -> torch.Tensor:
    """(D,3,32,32) -> (Cb, 32, 32, D) uint8 thermometer bit maps (Cb = 3*num_bits)."""
    b = enc(images)                                            # (D, Cb, 32, 32)
    return b.permute(1, 2, 3, 0).contiguous().to(torch.uint8)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/cifar-10-batches-py"))
    p.add_argument("--download", action="store_true")
    p.add_argument("--train-size", type=int, default=0)
    p.add_argument("--num-bits", type=int, default=5)
    p.add_argument("--radius", type=int, default=1, help="receptive field radius (1 = 3x3)")
    p.add_argument("--conv-channels", type=int, default=64, help="conv output channels M")
    p.add_argument("--conv-fan-in", type=int, default=4)
    p.add_argument("--conv-gates", type=int, default=4000, help="conv build-ops (rebuilds)")
    p.add_argument("--conv-build", type=int, default=64, help="conv channels (re)built per phase")
    p.add_argument("--conv-cd", type=int, default=2000, help="conv CD flips per phase")
    p.add_argument("--build-batch", type=int, default=256)
    p.add_argument("--cd-batch", type=int, default=4096)
    p.add_argument("--dense", action="store_true", help="run phase 2 dense on the conv output")
    p.add_argument("--window-factor", type=float, default=2.0)
    p.add_argument("--dense-fan-in", type=int, default=4)
    p.add_argument("--dense-gates", type=int, default=100000)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dev = args.device
    print(f"device={dev}  args={vars(args)}", flush=True)
    tx, ty, ex, ey = load_cifar10(args.data_dir, args.download)
    wx, wy = (tx, ty) if args.train_size == 0 else (tx[:args.train_size], ty[:args.train_size])
    nv = max(1, round(len(wx) * 0.1))
    vx, vy, px, py = wx[-nv:], wy[-nv:], wx[:-nv], wy[:-nv]
    print(f"train={len(px)} val={len(vx)} test={len(ex)}", flush=True)

    enc = Thermometer(num_bits=args.num_bits).fit(px[:2000])
    Mtr, Mva, Mte = encode_maps(px, enc), encode_maps(vx, enc), encode_maps(ex, enc)
    cb = Mtr.shape[0]
    ytr = py.to(dev)
    print(f"Cb={cb}  HW=32  conv: M={args.conv_channels} R={args.radius} K={args.conv_fan_in}", flush=True)

    conv = ConvLayer(cb, 32, 10, channels=args.conv_channels, radius=args.radius,
                     fan_in=args.conv_fan_in, device=dev)
    conv.set_inputs(Mtr)

    def rb(n):
        return torch.randint(conv.D, (min(n, conv.D),), device=dev)

    t0 = time.time()
    print("\nphase 1: conv\n  ph | built |  cv_tr | cv_va |  time", flush=True)
    ph = 0
    while ph * args.conv_build < args.conv_gates:
        ph += 1
        conv.build_sweep(ytr, rb(args.build_batch), args.conv_build)
        for _ in range(max(1, args.conv_cd // 256)):
            conv.cd_pass(ytr, rb(args.cd_batch), 256)
        if ph % args.eval_every == 0:
            print(f"{ph:4d} | {int(conv.built.sum()):5d} | {conv.evaluate(Mtr, py):6.2f} | "
                  f"{conv.evaluate(Mva, vy):6.2f} | {time.time() - t0:4.0f}s", flush=True)
    print(f"conv done: train={conv.evaluate(Mtr, py):.2f} val={conv.evaluate(Mva, vy):.2f} "
          f"test={conv.evaluate(Mte, ey):.2f}", flush=True)

    if not args.dense:
        return

    print("\nphase 2: dense on conv output", flush=True)
    Xtr = conv.transform(Mtr).cpu()                            # (N', D) dense input bits
    Xva, Xte = conv.transform(Mva).cpu(), conv.transform(Mte).cpu()
    nprime = Xtr.shape[0]
    print(f"dense input N' = M*H*W = {nprime}", flush=True)
    circ = GrownCircuit(nprime, 10, args.window_factor, fan_in=args.dense_fan_in,
                        max_gates=args.dense_gates, device=dev)
    circ.set_inputs(Xtr)
    best = -1.0
    pn = 0
    while circ.n_gates_built < args.dense_gates:
        pn += 1
        circ.build_sweep(ytr, rb(2000), 2000)
        target = int(0.5 * circ.WIN)
        done = 0
        while done < target:
            nf = min(2048, target - done)
            circ.cd_pass(ytr, rb(args.cd_batch), nf); done += nf
        if pn % args.eval_every == 0:
            va = circ.evaluate(Xva, vy)
            best = max(best, va)
            print(f"d{pn:3d} | gates {circ.n_gates_built:7d} | tr {circ.evaluate(Xtr, py):6.2f} | "
                  f"va {va:6.2f} | te {circ.evaluate(Xte, ey):6.2f} | {time.time() - t0:4.0f}s",
                  flush=True)
    print(f"\nFINAL dense: best_val={best:.2f}", flush=True)


if __name__ == "__main__":
    main()
