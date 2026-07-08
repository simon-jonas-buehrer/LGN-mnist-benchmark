"""Tree DOUBLE DESCENT (Belkin et al. 2019, the random-forest version).

The boosting capacity sweep (ddescent.sbatch) shows only the CLASSICAL half: val error falls,
then rises as a single model overfits past the interpolation threshold. To see the SECOND
descent -- val error falling AGAIN in the over-parameterised regime -- you need ensemble
averaging, not more capacity in one model. This script stitches the two regimes on ONE
#parameters axis (x = total #leaves in the ensemble ~ #parameters; y = error = 100 - accuracy):

  seg1  CLASSICAL (ensemble = 1): ONE tree, grow max_leaves 2 -> N. Error falls to the
        interpolation PEAK, where a single tree just fits the (subsampled) train set.
  seg2  MODERN   (ensemble > 1): FULLY-GROWN interpolating trees (min_leaf 1), grow the FOREST
        size 1 -> K by bagging (bootstrap rows + feature subsample), majority vote. Each tree
        memorises its bootstrap sample; averaging many of them drives val error back DOWN.

Train is subsampled (default 8000) so interpolation is cheap and the params/samples ratio is
clean. Reuses build_tree/predict/build_features from boost.py. Writes dd_curve.jsonl and, if
matplotlib is present, double_descent.png.

    python tree_scratch/dd.py --out tree_scratch/runs/dd
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Thermometer  # noqa: E402
from train import load_cifar10  # noqa: E402
from tree_scratch.boost import CLS, build_features, build_tree, predict  # noqa: E402


def n_leaves(tree: dict) -> int:
    return int((tree["feat"] < 0).sum())


def acc(pred: torch.Tensor, y: torch.Tensor) -> float:
    return 100.0 * float((pred == y).float().mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/cifar-10-batches-py"))
    p.add_argument("--download", action="store_true")
    p.add_argument("--num-bits", type=int, default=4)
    p.add_argument("--n-train", type=int, default=8000, help="subsample train (cheap interpolation)")
    p.add_argument("--max-features", type=int, default=2048, help="bits sampled per tree")
    p.add_argument("--seg1-leaves", type=str,
                   default="2,4,8,16,32,64,128,256,512,1024,2048,4096,8000",
                   help="single-tree capacities for the classical regime")
    p.add_argument("--forest", type=int, default=60, help="max forest size for the modern regime")
    p.add_argument("--min-leaf", type=int, default=1, help="1 => trees can fully interpolate")
    p.add_argument("--max-depth", type=int, default=100)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    dev = args.device
    g = torch.Generator(device="cpu").manual_seed(args.seed)

    # ---- data + lean binary features (subsample train) -----------------------------------
    tx, ty, ex, ey = load_cifar10(args.data_dir, args.download)
    nv = 5000
    enc32 = Thermometer(num_bits=args.num_bits).fit(tx[:2000]).to(dev)

    @torch.no_grad()
    def encode(images):
        return torch.cat([build_features(images[i:i + 4096].to(dev), enc32)
                          for i in range(0, len(images), 4096)], 0)

    trx, trY = tx[:-nv], ty[:-nv]
    sel = torch.randperm(len(trx), generator=g)[:args.n_train]
    Xtr, ytr = encode(trx[sel]), trY[sel].to(dev)
    Xva, yva = encode(tx[-nv:]), ty[-nv:].to(dev)
    Xte, yte = encode(ex), ey.to(dev)
    N, I = Xtr.shape
    MF = args.max_features or I
    allf = torch.arange(I, device=dev)
    w = torch.full((N,), 1.0 / N, device=dev)
    yoh = torch.zeros(N, CLS, device=dev)
    yoh[torch.arange(N, device=dev), ytr] = 1.0

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = out.with_suffix(".jsonl")
    jsonl.write_text("")
    print(f"dd N_train={N} I={I} MF={MF} seg1={args.seg1_leaves} forest={args.forest} "
          f"device={dev}", flush=True)

    def log(rec):
        with open(jsonl, "a") as f:
            f.write(json.dumps(rec) + "\n")

    t0 = time.time()
    # ---- seg1 CLASSICAL: one tree, growing capacity ---------------------------------------
    for L in (int(x) for x in args.seg1_leaves.split(",")):
        tree = build_tree(Xtr, allf, ytr, w, yoh * w[:, None], L, args.min_leaf, args.max_depth)
        lv = n_leaves(tree)
        rec = {"regime": "classical", "params": lv, "leaves": lv, "trees": 1,
               "train": round(acc(predict(tree, Xtr), ytr), 2),
               "val": round(acc(predict(tree, Xva), yva), 2),
               "test": round(acc(predict(tree, Xte), yte), 2),
               "min": round((time.time() - t0) / 60, 2)}
        log(rec)
        print(f"[classical] leaves {lv:6d} | train {rec['train']:6.2f} val {rec['val']:6.2f} "
              f"test {rec['test']:6.2f} | {rec['min']:5.2f}m", flush=True)

    # ---- seg2 MODERN: bagged fully-grown interpolating trees, growing forest size ---------
    votes = {"train": torch.zeros(N, CLS, device=dev),
             "val": torch.zeros(len(yva), CLS, device=dev),
             "test": torch.zeros(len(yte), CLS, device=dev)}
    sets = {"train": (Xtr, ytr), "val": (Xva, yva), "test": (Xte, yte)}
    cum_leaves = 0
    for k in range(1, args.forest + 1):
        boot = torch.randint(0, N, (N,), generator=g).to(dev)          # bootstrap rows
        gfeat = torch.randperm(I, generator=g)[:MF].sort().values.to(dev)
        wb = torch.full((N,), 1.0 / N, device=dev)
        ybh = torch.zeros(N, CLS, device=dev)
        ybh[torch.arange(N, device=dev), ytr[boot]] = 1.0
        tree = build_tree(Xtr[boot][:, gfeat], gfeat, ytr[boot], wb, ybh * wb[:, None],
                          N, args.min_leaf, args.max_depth)          # max_leaves=N => interpolate
        cum_leaves += n_leaves(tree)
        rec = {"regime": "modern", "params": cum_leaves, "leaves": n_leaves(tree), "trees": k,
               "min": round((time.time() - t0) / 60, 2)}
        for name, (X, yv) in sets.items():
            s = votes[name]
            s[torch.arange(s.shape[0], device=dev), predict(tree, X)] += 1.0
            rec[name] = round(acc(s.argmax(1), yv), 2)
        log(rec)
        print(f"[modern]  trees {k:3d} params {cum_leaves:8d} | train {rec['train']:6.2f} "
              f"val {rec['val']:6.2f} test {rec['test']:6.2f} | {rec['min']:5.2f}m", flush=True)

    print(f"done in {(time.time() - t0) / 60:.2f}m -> {jsonl}", flush=True)


if __name__ == "__main__":
    main()
