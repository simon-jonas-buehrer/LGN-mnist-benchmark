"""Grokking run: build a large net, STOP building, then run CD ~forever and watch for a late
val jump after train saturates.

Two efficiency tricks make "CD forever" feasible:
  - train accuracy is read straight from the maintained class scores (argmax of self.score) -- no
    replay, instant even after hundreds of millions of CD flips;
  - val/test eval replays the BUILD STRUCTURE ONCE with the CURRENT params (CD only changes params,
    not structure), so its cost is constant instead of growing with every kept flip.

CD uses a large/full batch so a flip is kept only if it lowers the FULL-train hinge (reliable,
~monotonic train improvement). We do NOT stop at the train target -- we note it and keep going,
sampling val regularly to catch grokking.

    .venv/bin/python scratch/fit_train.py --device cuda --window-factor 32 --cd-batch 45000
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
from grow_lut import GrownCircuit, pack_bits, unpack_bits  # noqa: E402


def encode(images, enc):
    return enc(images).flatten(1).t().contiguous().to(torch.uint8)


@torch.no_grad()
def train_acc(circ, y):
    """Instant train accuracy from the maintained class scores."""
    return 100.0 * (circ.score.argmax(0) == y).float().mean().item()


@torch.no_grad()
def fast_eval(circ, build_ops, Xbits, y, batch=8192):
    """Val/test accuracy: replay the build structure ONCE with current params (constant cost)."""
    d = Xbits.shape[1]
    correct = 0
    for i in range(0, d, batch):
        xb = Xbits[:, i:i + batch].to(circ.device)
        win = pack_bits(xb)[circ.tile].contiguous()
        for slots, ins, _ in build_ops:
            win[slots] = circ.apply_full(win[ins], circ.def_p[slots])   # current params, not stored
        score = torch.zeros((circ.C, xb.shape[1]), device=circ.device)
        for s0 in range(0, circ.WIN, 8192):
            sl = slice(s0, min(s0 + 8192, circ.WIN))
            b = unpack_bits(win[sl], xb.shape[1]).to(torch.float32)
            score.index_add_(0, circ.class_of[sl], b)
        correct += (score.argmax(0).cpu() == y[i:i + batch]).sum().item()
    return 100.0 * correct / d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/cifar-10-batches-py"))
    p.add_argument("--num-bits", type=int, default=5)
    p.add_argument("--window-factor", type=float, default=32.0)
    p.add_argument("--fan-in", type=int, default=4)
    p.add_argument("--build-batch", type=int, default=4096)
    p.add_argument("--build-per-phase", type=int, default=8000)
    p.add_argument("--cd-batch", type=int, default=45000)
    p.add_argument("--cd-flips", type=int, default=2048)
    p.add_argument("--target-train", type=float, default=95.0)
    p.add_argument("--max-flips", type=int, default=2_000_000_000)
    p.add_argument("--report-flips", type=int, default=4_000_000)
    p.add_argument("--val-every", type=int, default=20_000_000)
    p.add_argument("--aug-flip", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    dev = args.device
    print(f"args={vars(args)}", flush=True)
    tx, ty, ex, ey = load_cifar10(args.data_dir, False)
    nv = round(len(tx) * 0.1)
    vx, vy, px, py = tx[-nv:], ty[-nv:], tx[:-nv], ty[:-nv]
    if args.aug_flip:
        px, py = torch.cat([px, px.flip(-1)]), torch.cat([py, py])
    enc = Thermometer(num_bits=args.num_bits).fit(px[:2000])
    Xtr, Xva, Xte = encode(px, enc), encode(vx, enc), encode(ex, enc)
    n = Xtr.shape[0]
    ytr = py.to(dev)

    circ = GrownCircuit(n, 10, args.window_factor, fan_in=args.fan_in, gate_type="lut",
                        max_gates=10 ** 9, device=dev)
    circ.set_inputs(Xtr)
    print(f"N={n} WIN={circ.WIN} (f={args.window_factor}) building then STOPPING build...", flush=True)
    t0 = time.time()
    while circ.n_gates_built < circ.WIN:
        circ.build_sweep(ytr, torch.randint(circ.D, (args.build_batch,), device=dev),
                         args.build_per_phase, depth_pen=2.0, usage_pen=0.3)
    build_ops = list(circ.ops)                       # structure snapshot for fast eval
    print(f"built {circ.n_gates_built} gates in {time.time()-t0:.0f}s  "
          f"train={train_acc(circ, ytr):.2f}  val={fast_eval(circ, build_ops, Xva, vy):.2f}\n",
          flush=True)

    print(f"{'flips':>12} | {'train':>6} | {'val':>6} | {'test':>6} | {'kfl/s':>6} | note", flush=True)
    flips, last_val, hit = 0, 0, False
    while flips < args.max_flips:
        tc = time.time()
        d2 = 0
        while d2 < args.report_flips:
            nf = min(args.cd_flips, args.report_flips - d2)
            circ.cd_pass(ytr, torch.randint(circ.D, (args.cd_batch,), device=dev), nf)
            d2 += nf
        flips += args.report_flips
        tr = train_acc(circ, ytr)
        rate = args.report_flips / (time.time() - tc) / 1000
        va = te = float("nan")
        note = ""
        if flips - last_val >= args.val_every:
            last_val = flips
            va, te = fast_eval(circ, build_ops, Xva, vy), fast_eval(circ, build_ops, Xte, ey)
        if tr >= args.target_train and not hit:
            hit = True
            note = f"<-- train target {args.target_train} reached; continuing CD for grokking"
        print(f"{flips:>12} | {tr:6.2f} | {va:6.2f} | {te:6.2f} | {rate:6.1f} | {note}", flush=True)
    print(f"\nDONE flips={flips} train={train_acc(circ, ytr):.2f}", flush=True)


if __name__ == "__main__":
    main()
