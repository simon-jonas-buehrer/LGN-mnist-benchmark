"""tree_scratch: CONV-TREE layer + boosted-tree head (a convolutional deep forest).

The unification of the design thread: a decision tree used as a weight-shared CONVOLUTIONAL
filter (gcForest multi-grained scanning), OR-pooled, feeding the leaf-wise boosted-tree head
from boost.py. Every piece stays Boolean: each conv-tree is a DNF over its patch bits, applied
at every position (a shift-invariant DNF), OR-pool is literal OR, the head is a threshold of
DNFs. So the whole model is an extractable logic circuit -- the FPGA/LUT target -- with a whole
tree as the gate instead of a 2-input LUT.

Pipeline:
  image -> Thermometer bits (B, 3*nb, 32, 32)
        -> CONV-TREE layer: per grain K, train a boosted forest on labelled KxK patches
           (patch -> image label, weak per patch but informative pooled), apply it at every
           strided position -> per-position argmax one-hot (CLS channels) -> OR-pool 2x2
        -> concat grains (+ optionally the census/edge features from boost.build_features)
        -> leaf-wise SAMME boosted-tree HEAD -> class

This is one conv layer; stacking (apply another conv-tree layer on the CLS-channel bit maps)
is the natural next step once one layer is shown to help.

    python tree_scratch/conv_boost.py --out tree_scratch/runs/c0
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Thermometer  # noqa: E402
from train import load_cifar10  # noqa: E402
from tree_scratch.boost import build_features, build_tree, predict  # noqa: E402

CLS = 10


# ==========================================================================================
# Patch extraction and a compact SAMME boosting fit (reuses boost.build_tree/predict)
# ==========================================================================================
def patches(x: torch.Tensor, K: int, stride: int) -> torch.Tensor:
    """(b, C, H, W) uint8 -> (b, Ho, Wo, C*K*K): every KxK patch flattened, stride `stride`."""
    b, C, H, W = x.shape
    xp = x.unfold(2, K, stride).unfold(3, K, stride)      # (b, C, Ho, Wo, K, K)
    Ho, Wo = xp.shape[2], xp.shape[3]
    return xp.permute(0, 2, 3, 1, 4, 5).reshape(b, Ho, Wo, C * K * K)


def ensemble_scores(trees: list, alphas: list, X: torch.Tensor) -> torch.Tensor:
    """(B, CLS) weighted class votes of a boosted forest over rows of X."""
    s = torch.zeros(X.shape[0], CLS, device=X.device)
    ar = torch.arange(X.shape[0], device=X.device)
    for tr, a in zip(trees, alphas):
        s[ar, predict(tr, X)] += a
    return s


def fit_boost(X: torch.Tensor, y: torch.Tensor, *, n_trees: int, max_leaves: int, min_leaf: int,
              max_features: int, lr: float, seed: int, evalsets: dict | None = None,
              logpath: Path | None = None, tag: str = "") -> tuple[list, list]:
    """Leaf-wise SAMME boosting (the boost.py mechanism, packaged). Returns (trees, alphas),
    trees kept on-device. If evalsets given, logs train/val/test ensemble accuracy per tree."""
    dev = X.device
    N, I = X.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    yoh = torch.zeros(N, CLS, device=dev)
    yoh[torch.arange(N, device=dev), y] = 1.0
    w = torch.full((N,), 1.0 / N, device=dev)
    MF = max_features or I
    trees: list = []
    alphas: list = []
    score = ({n: torch.zeros(len(yy), CLS, device=dev) for n, (_, yy) in evalsets.items()}
             if evalsets else {})
    t0 = time.time()
    for t in range(n_trees):
        gfeat = (torch.randperm(I, generator=g)[:MF].sort().values.to(dev)
                 if MF < I else torch.arange(I, device=dev))
        tree = build_tree(X[:, gfeat], gfeat, y, w, yoh * w[:, None], max_leaves, min_leaf, 64)
        miss = (predict(tree, X) != y).to(torch.float32)
        err = float((w * miss).sum() / w.sum())
        if err >= 1.0 - 1.0 / CLS:
            w = torch.full((N,), 1.0 / N, device=dev)
            continue
        if err <= 1e-12:
            alpha = (math.log((1 - 1e-12) / 1e-12) + math.log(CLS - 1)) * lr
            w = torch.full((N,), 1.0 / N, device=dev)
        else:
            alpha = (math.log((1 - err) / err) + math.log(CLS - 1)) * lr
            w = (w * torch.exp(alpha * miss))
            w = w / w.sum()
        trees.append(tree)
        alphas.append(alpha)
        if evalsets and (t % 10 == 0 or t == n_trees - 1):
            rec = {"tree": t, "err": round(err, 4), "min": round((time.time() - t0) / 60, 2)}
            for n, (Xe, ye) in evalsets.items():
                s = score[n]
                s[torch.arange(s.shape[0], device=dev), predict(tree, Xe)] += alpha
                rec[n] = round(100.0 * float((s.argmax(1) == ye).float().mean()), 2)
            if logpath:
                with open(logpath, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            print(f"{tag}tree {t:4d} | err {err:.3f} | "
                  + " ".join(f"{n} {rec[n]:5.2f}" for n in evalsets) + f" | {rec['min']:5.2f}m",
                  flush=True)
        elif evalsets:                                    # keep running scores current between logs
            for n, (Xe, _) in evalsets.items():
                score[n][torch.arange(score[n].shape[0], device=dev), predict(tree, Xe)] += alpha
    return trees, alphas


# ==========================================================================================
# The conv-tree layer: fit forests on patches, apply them convolutionally + OR-pool
# ==========================================================================================
def fit_conv(Xmap: torch.Tensor, y: torch.Tensor, grains: list[int], args) -> list:
    """Train one boosted forest per grain on a subsample of labelled patches. Returns the
    per-grain (K, forest) so the same weight-shared trees can be applied to any image set."""
    dev = Xmap.device
    saved = []
    for gi, K in enumerate(grains):
        P = patches(Xmap, K, args.conv_stride)            # (B, Ho, Wo, dim)
        dim = P.shape[-1]
        Pf = P.reshape(-1, dim)
        yl = y[:, None, None].expand(-1, P.shape[1], P.shape[2]).reshape(-1)
        g = torch.Generator(device="cpu").manual_seed(args.seed + gi)
        n = min(args.conv_samples, Pf.shape[0])
        sel = torch.randperm(Pf.shape[0], generator=g)[:n].to(dev)
        print(f"  grain K={K}: {Pf.shape[0]:,} patches (dim {dim}), train on {n:,}", flush=True)
        trees, alphas = fit_boost(Pf[sel], yl[sel], n_trees=args.conv_trees,
                                  max_leaves=args.conv_leaves, min_leaf=args.conv_min_leaf,
                                  max_features=0, lr=1.0, seed=args.seed + gi)
        saved.append((K, trees, alphas))
    return saved


@torch.no_grad()
def apply_conv(Xmap: torch.Tensor, saved: list, args) -> torch.Tensor:
    """Apply the weight-shared conv-forests at every strided position -> per-position argmax
    one-hot (CLS channels) -> OR-pool 2x2 -> flatten; concat over grains. (B, F_conv) uint8."""
    dev = Xmap.device
    B = Xmap.shape[0]
    feats = []
    for K, trees, alphas in saved:
        P = patches(Xmap, K, args.conv_stride)            # (B, Ho, Wo, dim)
        Ho, Wo, dim = P.shape[1], P.shape[2], P.shape[3]
        Pf = P.reshape(-1, dim)
        pred = torch.empty(Pf.shape[0], dtype=torch.long, device=dev)
        for i in range(0, Pf.shape[0], args.apply_chunk):
            pred[i:i + args.apply_chunk] = ensemble_scores(trees, alphas,
                                                           Pf[i:i + args.apply_chunk]).argmax(1)
        oh = torch.zeros(Pf.shape[0], CLS, dtype=torch.uint8, device=dev)
        oh[torch.arange(Pf.shape[0], device=dev), pred] = 1
        fmap = oh.view(B, Ho, Wo, CLS).permute(0, 3, 1, 2)   # (B, CLS, Ho, Wo)
        fmap = F.max_pool2d(fmap.float(), 2, ceil_mode=True).to(torch.uint8)  # OR-pool
        feats.append(fmap.reshape(B, -1))
    return torch.cat(feats, 1)


# ==========================================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/cifar-10-batches-py"))
    p.add_argument("--num-bits", type=int, default=4)
    p.add_argument("--grains", type=str, default="3,5", help="conv receptive-field sizes")
    p.add_argument("--conv-stride", type=int, default=2)
    p.add_argument("--conv-trees", type=int, default=80)
    p.add_argument("--conv-leaves", type=int, default=64)
    p.add_argument("--conv-min-leaf", type=int, default=20)
    p.add_argument("--conv-samples", type=int, default=200_000, help="patches to train each grain")
    p.add_argument("--apply-chunk", type=int, default=400_000)
    p.add_argument("--with-census", type=int, default=1, help="concat boost.build_features edges")
    # head
    p.add_argument("--max-leaves", type=int, default=512)
    p.add_argument("--trees", type=int, default=1500)
    p.add_argument("--lr", type=float, default=0.3)
    p.add_argument("--min-leaf", type=int, default=4)
    p.add_argument("--max-features", type=int, default=4096)
    p.add_argument("--max-train", type=int, default=0, help="cap train images (0=all; for smoke)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    dev = args.device
    grains = [int(x) for x in args.grains.split(",")]

    tx, ty, ex, ey = load_cifar10(args.data_dir, False)
    nv = 5000
    sample = tx[:2000]
    enc32 = Thermometer(num_bits=args.num_bits).fit(sample).to(dev)
    enc16 = Thermometer(num_bits=args.num_bits).fit(F.avg_pool2d(sample, 2)).to(dev)
    trx, trY = tx[:-nv], ty[:-nv]
    if args.max_train:
        trx, trY = trx[:args.max_train], trY[:args.max_train]

    @torch.no_grad()
    def bitmap(images):                                   # thermometer bit image (B, 3*nb, 32, 32)
        return torch.cat([enc32(images[i:i + 4096].to(dev)).to(torch.uint8)
                          for i in range(0, len(images), 4096)], 0)

    @torch.no_grad()
    def census_feats(images):                             # boost.build_features, chunked
        return torch.cat([build_features(images[i:i + 4096].to(dev), enc32, enc16)
                          for i in range(0, len(images), 4096)], 0)

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".jsonl").write_text("")
    print(f"conv_boost grains={grains} stride={args.conv_stride} conv_trees={args.conv_trees} "
          f"leaves={args.conv_leaves} census={args.with_census} head_leaves={args.max_leaves} "
          f"trees={args.trees} lr={args.lr} train={len(trx)} device={dev}", flush=True)

    # --- conv-tree layer: fit on train patches, apply to every set --------------------------
    print("fitting conv-tree forests...", flush=True)
    saved = fit_conv(bitmap(trx), trY.to(dev), grains, args)

    def head_feats(images):
        m = bitmap(images)
        parts = [apply_conv(m, saved, args)]
        if args.with_census:
            parts.append(census_feats(images))
        return torch.cat(parts, 1)

    print("building head features...", flush=True)
    Xtr, ytr = head_feats(trx), trY.to(dev)
    Xva, yva = head_feats(tx[-nv:]), ty[-nv:].to(dev)
    Xte, yte = head_feats(ex), ey.to(dev)
    print(f"head feature dim = {Xtr.shape[1]} (conv+census)", flush=True)

    # --- boosted-tree head ------------------------------------------------------------------
    evalsets = {"train": (Xtr, ytr), "val": (Xva, yva), "test": (Xte, yte)}
    trees, alphas = fit_boost(Xtr, ytr, n_trees=args.trees, max_leaves=args.max_leaves,
                              min_leaf=args.min_leaf, max_features=args.max_features, lr=args.lr,
                              seed=args.seed, evalsets=evalsets, logpath=out.with_suffix(".jsonl"))
    with open(out.with_suffix(".pkl"), "wb") as f:
        pickle.dump({"args": vars(args) | {"out": str(out)}, "grains": grains,
                     "conv": [(K, [{k: v.cpu() for k, v in t.items()} for t in trs], al)
                              for K, trs, al in saved],
                     "head_alphas": alphas,
                     "head_trees": [{k: v.cpu() for k, v in t.items()} for t in trees]}, f)
    print(f"saved -> {out.with_suffix('.pkl')}", flush=True)


if __name__ == "__main__":
    main()
