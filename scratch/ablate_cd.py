"""Ablation: does CD help a FULLY-BUILT LARGE network, and what cd-batch is needed?

Earlier evidence that "CD degrades" was all on tiny nets (15k gates, depth ~0.5) or with CD
interleaved into building. The missed regime: build a large net to completion FIRST, then run
CD-only and watch whether it helps -- and how that depends on the CD batch size. The flip-accept
test in cd_pass uses the hinge on the *batch*, so a small batch accepts flips that overfit the
batch (hurting full train/val); a big batch should accept only real improvements.

We build the large net ONCE (no CD), snapshot it, then for each cd-batch restore the snapshot and
run the SAME extended CD budget, tracking train+val accuracy over the run.

    .venv/bin/python scratch/ablate_cd.py --device cuda --window-factor 8 --build-batch 4096 \
        --cd-flips 2048 --cd-total 3000000 --report-every 300000
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
from grow_lut import GrownCircuit  # noqa: E402


def encode(images, enc):
    return enc(images).flatten(1).t().contiguous().to(torch.uint8)


def augment(x, y, *, flip=True, crops=0, pad=4):
    """Expand the train set with augmented copies (original + h-flip + random crops). Returns the
    concatenated images and matching labels."""
    import torch.nn.functional as F
    parts = [(x, y)]
    if flip:
        parts.append((x.flip(-1), y))
    if crops > 0:
        xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")        # (N,3,32+2p,32+2p)
        nn_, cc = torch.arange(x.shape[0]).view(-1, 1, 1, 1), torch.arange(3).view(1, 3, 1, 1)
        ar = torch.arange(32)
        for _ in range(crops):
            i = torch.randint(0, 2 * pad + 1, (x.shape[0], 1, 1, 1))
            j = torch.randint(0, 2 * pad + 1, (x.shape[0], 1, 1, 1))
            ri = i + ar.view(1, 1, 32, 1)
            cj = j + ar.view(1, 1, 1, 32)
            parts.append((xp[nn_, cc, ri, cj], y))
    return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])


def snapshot_state(c):
    return (c.win.clone(), c.score.clone(), c.def_in.clone(), c.def_p.clone(),
            c.depth.clone(), c.usage.clone(), list(c.ops), c.n_gates_built)


def restore_state(c, s):
    c.win, c.score, c.def_in, c.def_p, c.depth, c.usage, ops, n = s
    c.win = c.win.clone(); c.score = c.score.clone(); c.def_in = c.def_in.clone()
    c.def_p = c.def_p.clone(); c.depth = c.depth.clone(); c.usage = c.usage.clone()
    c.ops = list(ops); c.n_gates_built = n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/cifar-10-batches-py"))
    p.add_argument("--num-bits", type=int, default=5)
    p.add_argument("--window-factor", type=float, default=8.0)
    p.add_argument("--fan-in", type=int, default=4)
    p.add_argument("--build-batch", type=int, default=4096)
    p.add_argument("--build-per-phase", type=int, default=4000)
    p.add_argument("--build-gates", type=int, default=0, help="0 = fill window once (WIN gates)")
    p.add_argument("--cd-flips", type=int, default=2048, help="flips attempted per cd_pass")
    p.add_argument("--cd-total", type=int, default=3000000, help="total flips attempted per config")
    p.add_argument("--report-every", type=int, default=300000)
    p.add_argument("--cd-batches", type=int, nargs="+", default=[2048, 8192, 32768, 45000])
    p.add_argument("--aug-flip", action="store_true", help="add horizontal-flip copies of train")
    p.add_argument("--aug-crops", type=int, default=0, help="add this many random-crop copies")
    p.add_argument("--max-train", type=int, default=0, help="subsample (augmented) train to this size")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    dev = args.device
    print(f"args={vars(args)}", flush=True)
    tx, ty, ex, ey = load_cifar10(args.data_dir, False)
    nv = round(len(tx) * 0.1)
    vx, vy, px, py = tx[-nv:], ty[-nv:], tx[:-nv], ty[:-nv]
    if args.aug_flip or args.aug_crops:
        px, py = augment(px, py, flip=args.aug_flip, crops=args.aug_crops)
        if args.max_train and len(px) > args.max_train:           # cap to bound memory
            sel = torch.randperm(len(px))[:args.max_train]
            px, py = px[sel], py[sel]
        print(f"augmented train -> {len(px)} images (flip={args.aug_flip} crops={args.aug_crops})",
              flush=True)
    enc = Thermometer(num_bits=args.num_bits).fit(px[:2000])
    Xtr, Xva = encode(px, enc), encode(vx, enc)
    n = Xtr.shape[0]
    ytr = py.to(dev)

    circ = GrownCircuit(n, 10, args.window_factor, fan_in=args.fan_in, gate_type="lut",
                        max_gates=10 ** 9, device=dev)
    circ.set_inputs(Xtr)
    target = args.build_gates or circ.WIN
    print(f"N={n} WIN={circ.WIN} building large net to {target} gates (no CD)...", flush=True)
    t0 = time.time()
    while circ.n_gates_built < target:
        circ.build_sweep(ytr, torch.randint(circ.D, (args.build_batch,), device=dev),
                         args.build_per_phase, depth_pen=2.0, usage_pen=0.3)
    base_tr, base_va = circ.evaluate(Xtr, py), circ.evaluate(Xva, vy)
    md = circ.depth.float().mean().item()
    print(f"built {circ.n_gates_built} gates in {time.time()-t0:.0f}s  depth_mean={md:.2f}  "
          f"BEFORE-CD train={base_tr:.2f} val={base_va:.2f}\n", flush=True)
    snap = snapshot_state(circ)

    print(f"{'cd_batch':>8} | {'flips':>9} | {'train':>6} | {'val':>6} | note", flush=True)
    print(f"{'(start)':>8} | {0:>9} | {base_tr:6.2f} | {base_va:6.2f} |", flush=True)
    for cb in args.cd_batches:
        restore_state(circ, snap)
        done, best_va = 0, base_va
        t1 = time.time()
        while done < args.cd_total:
            target_chunk = min(args.report_every, args.cd_total - done)
            d2 = 0
            while d2 < target_chunk:
                nf = min(args.cd_flips, target_chunk - d2)
                circ.cd_pass(ytr, torch.randint(circ.D, (cb,), device=dev), nf)
                d2 += nf
            done += target_chunk
            tr, va = circ.evaluate(Xtr, py), circ.evaluate(Xva, vy)
            best_va = max(best_va, va)
            print(f"{cb:>8} | {done:>9} | {tr:6.2f} | {va:6.2f} | {time.time()-t1:4.0f}s", flush=True)
        print(f"{cb:>8} |  -> best val={best_va:.2f}  (delta {best_va-base_va:+.2f})\n", flush=True)


if __name__ == "__main__":
    main()
