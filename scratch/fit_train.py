"""Push TRAIN accuracy as high as possible (target ~95%) on the LUT-network classifier.

Fitting, not generalization. Two things make this efficient:
  - train accuracy is read straight from the maintained class scores (argmax of self.score over the
    train set) -- no op-replay, so it's instant even after tens of millions of CD flips;
  - CD uses a LARGE/full batch so a flip is kept only if it lowers the FULL-train hinge -> train
    improves ~monotonically instead of plateauing on batch noise.

Build a large net (big f), then CD-only until train hits the target or the flip budget runs out.
Val/test are evaluated only occasionally (those DO need the slow replay).

    .venv/bin/python scratch/fit_train.py --device cuda --window-factor 32 --cd-batch 45000 \
        --target-train 95 --max-flips 200000000
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
from grow_lut import GrownCircuit, unpack_bits  # noqa: E402


def encode(images, enc):
    return enc(images).flatten(1).t().contiguous().to(torch.uint8)


@torch.no_grad()
def train_acc_from_score(circ, y):
    """Instant train accuracy from the maintained class scores (no replay)."""
    return 100.0 * (circ.score.argmax(0) == y).float().mean().item()


@torch.no_grad()
def train_acc_from_win(circ, y, chunk=8192):
    """Exact train accuracy recomputed from self.win (group-sum by class). Used to check score drift."""
    score = torch.zeros((circ.C, circ.D), device=circ.device)
    for s0 in range(0, circ.WIN, chunk):
        sl = slice(s0, min(s0 + chunk, circ.WIN))
        b = unpack_bits(circ.win[sl], circ.D).to(torch.float32)
        score.index_add_(0, circ.class_of[sl], b)
    return 100.0 * (score.argmax(0) == y).float().mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/cifar-10-batches-py"))
    p.add_argument("--num-bits", type=int, default=5)
    p.add_argument("--window-factor", type=float, default=32.0)
    p.add_argument("--fan-in", type=int, default=4)
    p.add_argument("--build-batch", type=int, default=4096)
    p.add_argument("--build-per-phase", type=int, default=8000)
    p.add_argument("--cd-batch", type=int, default=45000, help="full train by default (reliable flips)")
    p.add_argument("--cd-flips", type=int, default=2048)
    p.add_argument("--target-train", type=float, default=95.0)
    p.add_argument("--max-flips", type=int, default=300_000_000)
    p.add_argument("--report-flips", type=int, default=2_000_000)
    p.add_argument("--val-every", type=int, default=40_000_000, help="full val/test replay (slow)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    dev = args.device
    print(f"args={vars(args)}", flush=True)
    tx, ty, ex, ey = load_cifar10(args.data_dir, False)
    nv = round(len(tx) * 0.1)
    vx, vy, px, py = tx[-nv:], ty[-nv:], tx[:-nv], ty[:-nv]
    enc = Thermometer(num_bits=args.num_bits).fit(px[:2000])
    Xtr, Xva, Xte = encode(px, enc), encode(vx, enc), encode(ex, enc)
    n = Xtr.shape[0]
    ytr = py.to(dev)

    circ = GrownCircuit(n, 10, args.window_factor, fan_in=args.fan_in, gate_type="lut",
                        max_gates=10 ** 9, device=dev)
    circ.set_inputs(Xtr)
    print(f"N={n} WIN={circ.WIN} (f={args.window_factor}) building...", flush=True)
    t0 = time.time()
    while circ.n_gates_built < circ.WIN:
        circ.build_sweep(ytr, torch.randint(circ.D, (args.build_batch,), device=dev),
                         args.build_per_phase, depth_pen=2.0, usage_pen=0.3)
    print(f"built {circ.n_gates_built} gates in {time.time()-t0:.0f}s  "
          f"train(score)={train_acc_from_score(circ, ytr):.2f}\n", flush=True)

    print(f"{'flips':>11} | {'train':>6} | {'flips/s':>8} | note", flush=True)
    flips = 0
    last_val = 0
    t1 = time.time()
    while flips < args.max_flips:
        d2 = 0
        tc = time.time()
        while d2 < args.report_flips:
            nf = min(args.cd_flips, args.report_flips - d2)
            circ.cd_pass(ytr, torch.randint(circ.D, (args.cd_batch,), device=dev), nf)
            d2 += nf
        flips += args.report_flips
        tr = train_acc_from_score(circ, ytr)
        rate = args.report_flips / (time.time() - tc)
        note = ""
        if flips - last_val >= args.val_every:
            last_val = flips
            note = (f"val={circ.evaluate(Xva, vy):.2f} test={circ.evaluate(Xte, ey):.2f} "
                    f"train_exact={train_acc_from_win(circ, ytr):.2f}")
        print(f"{flips:>11} | {tr:6.2f} | {rate:8.0f} | {note}", flush=True)
        if tr >= args.target_train:
            print(f"\nREACHED target train {args.target_train} at {flips} flips "
                  f"({time.time()-t1:.0f}s)", flush=True)
            print(f"val={circ.evaluate(Xva, vy):.2f} test={circ.evaluate(Xte, ey):.2f} "
                  f"train_exact={train_acc_from_win(circ, ytr):.2f}", flush=True)
            break
    print(f"\nDONE flips={flips} train={train_acc_from_score(circ, ytr):.2f}", flush=True)


if __name__ == "__main__":
    main()
