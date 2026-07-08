"""tree_scratch: a BOOSTED single-bit decision-tree forest over the thermometer bits.

The dense method of the tree line (the conv variant is conv_boost.py, which stacks a conv-tree
layer under a boosted head reusing fit_boost from here). Deep single-bit decision trees built
one at a time on the CURRENT MISTAKES and combined by CONFIDENCE-WEIGHTED VOTING -- multiclass
AdaBoost (SAMME). Why boosting and not a flat popcount vote:

  * MISTAKE-CHASING. Each tree carries per-sample WEIGHTS; after a tree is built we UP-WEIGHT the
    samples it got wrong so the next tree is pushed to fix them. A plain equal-vote forest is
    blind to what the ensemble already gets right/wrong; boosting spends new capacity exactly on
    the errors. In multiclass every misclassification (whatever the confusion) is up-weighted
    symmetrically -- the false-positive-AND-false-negative buffer in one number.

  * COMPETENCE-WEIGHTED VOTE. alpha_t = ln((1-e_t)/e_t) + ln(K-1) (SAMME). A coin-flip tree
    (e = 1-1/K) gets alpha = 0 and is silenced; a good tree votes loud. Readout is
    argmax_c sum_t alpha_t * 1[tree_t(x) = c] -- a weighted-majority threshold-of-DNFs circuit.
    VOTE, not OR: OR-of-trees is monotone (fixes only false negatives); a signed weighted vote
    pushes both ways.

  * SINGLE-BIT SPLITS => BOOLEAN. Each internal node tests one bit x[f]; a root->leaf path is a
    CONJUNCTION of literals, so a tree is a DNF over the bits (export_rules reads it off). A
    depth-d tree already expresses any d-way interaction -- depth is the knob, no parity/monomial
    split features needed.

  * DEEP trees are strong learners, which would make classic AdaBoost degenerate (a zero-error
    tree => alpha -> inf, ensemble stalls). Two things keep "many trees on mistakes" alive:
    per-tree FEATURE SUBSAMPLING (--max-features) keeps trees diverse and their weighted error
    > 0, and a perfect tree RESETS the weights so the next tree explores a fresh subspace.

  * LEAF-WISE growth with a --max-leaves budget (split the leaf whose split cuts impurity most,
    wherever it sits) -- capacity lands only where the data needs a decision; #leaves = #DNF
    terms stays bounded. Small budgets = weak learners = the healthy boosting regime.

Lean binary features only (thermometer + per-pixel census edges + color-order; NO pooling, NO
convolution -- that lives in conv_boost.py). Self-contained (torch + repo encoder/data),
device-agnostic (CPU here, CUDA when present). No augmentation -- boosting keeps per-sample
weights over a FIXED train set, which augmentation would break.

    python tree_scratch/boost.py --out tree_scratch/runs/b0
"""

from __future__ import annotations

import argparse
import heapq
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

CLS = 10
NEG_INF = float("-inf")

# Census offsets (radius <= 2): each is an edge-sign bit `pixel > neighbor`, per channel.
OFFS = [(0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0), (2, 2), (2, -2)]


def census(img: torch.Tensor, offsets: list[tuple[int, int]], R: int) -> torch.Tensor:
    """(B,C,H,W) real -> (B, C*len(offsets), H, W) uint8 edge-sign bits `img > shift(img)`.
    Replicate padding so borders don't wrap. Illumination-invariant local gradients."""
    B, C, H, W = img.shape
    p = F.pad(img, (R, R, R, R), mode="replicate")
    outs = [(img > p[:, :, R + dy:R + dy + H, R + dx:R + dx + W]).to(torch.uint8)
            for dy, dx in offsets]
    return torch.cat(outs, 1)


def augment_train(imgs: torch.Tensor, labels: torch.Tensor, *, hflip: bool, crops: int,
                  pad: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Grow a FIXED augmented train set (extra ROWS, not on-the-fly -- boosting keeps per-sample
    weights over a fixed set). original (+ horizontal flip) (+ `crops` per-image random reflect-pad
    crops of radius `pad`). The single biggest accuracy lever on CIFAR for this model class."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    N, C, H, W = imgs.shape
    outs, ys = [imgs], [labels]
    if hflip:
        outs.append(torch.flip(imgs, dims=[3]))
        ys.append(labels)
    ar_h, ar_w = torch.arange(H), torch.arange(W)
    for _ in range(crops):
        p = F.pad(imgs, (pad, pad, pad, pad), mode="reflect")        # (N,C,H+2p,W+2p)
        oy = torch.randint(0, 2 * pad + 1, (N,), generator=gen)      # per-image crop offsets
        ox = torch.randint(0, 2 * pad + 1, (N,), generator=gen)
        ridx = (oy[:, None] + ar_h)[:, None, :, None].expand(N, C, H, W + 2 * pad)
        pr = p.gather(2, ridx)                                       # pick 32 rows per image
        cidx = (ox[:, None] + ar_w)[:, None, None, :].expand(N, C, H, W)
        outs.append(pr.gather(3, cidx))                             # pick 32 cols per image
        ys.append(labels)
    return torch.cat(outs, 0), torch.cat(ys, 0)


def build_features(imgs: torch.Tensor, enc32: Thermometer) -> torch.Tensor:
    """Lean binary feature map -> (B, F) uint8 bits, all Boolean so the ensemble stays
    extractable. Deliberately NO pooling / NO convolution: thermometer intensity bits,
    per-pixel census edge-sign bits (cheap locality), and color-order bits."""
    cen = census(imgs, OFFS, 2)                            # (B, C*8, 32, 32) edge signs
    r, g, b = imgs[:, 0:1], imgs[:, 1:2], imgs[:, 2:3]
    feats = [
        enc32(imgs).flatten(1),                            # thermometer intensity @ 32x32
        cen.flatten(1),                                    # edge signs @ 32x32  (locality)
        torch.cat([r > g, g > b, r > b], 1).flatten(1),    # color-order bits
    ]
    return torch.cat([f.to(torch.uint8) for f in feats], 1)


# ==========================================================================================
# One deep single-bit decision tree, built greedily on WEIGHTED samples (Gini). Four parallel
# arrays over nodes: feat (GLOBAL bit index, -1 at a leaf), left, right (child ids), cls (leaf
# weighted-majority class). Splits are chosen on the tree's feature subset; feat stores the
# GLOBAL id so prediction/extraction need only the full bit matrix.
# ==========================================================================================
def build_tree(Xsub: torch.Tensor, gfeat: torch.Tensor, y: torch.Tensor, w: torch.Tensor,
               wyoh: torch.Tensor, max_leaves: int, min_leaf: int, max_depth: int) -> dict:
    """LEAF-WISE (best-first): repeatedly split the leaf whose split most cuts weighted Gini,
    until the leaf budget is spent. max_depth is a safety guard.

    Xsub (N, F) uint8 bits on the tree's subsampled features; gfeat (F,) their global ids;
    y (N,) long; w (N,) weights; wyoh (N, K) = w in the sample's class column else 0."""
    dev = Xsub.device
    K = wyoh.shape[1]
    feat: list[int] = []
    left: list[int] = []
    right: list[int] = []
    cls: list[int] = []

    def add_leaf(idx: torch.Tensor):                     # append a leaf, return (node_id, cw)
        cw = torch.zeros(K, device=dev).index_add_(0, y[idx], w[idx])
        nid = len(feat)
        feat.append(-1)
        left.append(-1)
        right.append(-1)
        cls.append(int(cw.argmax()))
        return nid, cw

    def best_split(idx: torch.Tensor, cw: torch.Tensor):  # -> (gain, gfeat, lidx, ridx) | None
        Xg = Xsub[idx].to(torch.float32)                # (n, F); cast only this subset (memory)
        b1 = wyoh[idx].t() @ Xg                          # (K, F) weighted count of bit==1
        raw1 = Xg.sum(0)                                 # (F,)   raw count of bit==1
        n = idx.numel()
        cntR = b1                                        # bit==1 goes right
        cntL = cw[:, None] - b1                          # bit==0 goes left
        valid = (raw1 >= min_leaf) & (n - raw1 >= min_leaf)
        # Gini surrogate: maximise sum_c cnt^2 / n_side (= minimise weighted Gini impurity).
        score = (cntL.pow(2).sum(0) / cntL.sum(0).clamp_min(1e-12)
                 + cntR.pow(2).sum(0) / cntR.sum(0).clamp_min(1e-12))
        score = torch.where(valid, score, torch.full_like(score, NEG_INF))
        base = cw.pow(2).sum() / cw.sum().clamp_min(1e-12)
        best = int(score.argmax())
        gain = float(score[best]) - float(base)
        if not bool(valid[best]) or gain <= 1e-9:        # no split helps -> stays a leaf
            return None
        bit = Xg[:, best] > 0
        return gain, int(gfeat[best]), idx[~bit], idx[bit]

    heap: list = []                                      # best-first frontier: (-gain, ctr, ...)
    ctr = 0

    def consider(nid: int, idx: torch.Tensor, cw: torch.Tensor, depth: int):
        nonlocal ctr
        if depth >= max_depth or idx.numel() < 2 * min_leaf:
            return
        bs = best_split(idx, cw)
        if bs is not None:
            heapq.heappush(heap, (-bs[0], ctr, nid, idx, depth, bs))
            ctr += 1

    root_idx = torch.arange(Xsub.shape[0], device=dev)
    root, cw0 = add_leaf(root_idx)
    consider(root, root_idx, cw0, 0)
    leaves = 1
    while heap and leaves < max_leaves:
        _, _, nid, idx, depth, bs = heapq.heappop(heap)
        _, gf, lidx, ridx = bs
        lc, lcw = add_leaf(lidx)
        rc, rcw = add_leaf(ridx)
        feat[nid], left[nid], right[nid] = gf, lc, rc    # the popped leaf becomes internal
        leaves += 1                                      # -1 (nid) + 2 (children) = +1 leaf
        consider(lc, lidx, lcw, depth + 1)
        consider(rc, ridx, rcw, depth + 1)

    return {"feat": torch.tensor(feat, device=dev),
            "left": torch.tensor(left, device=dev),
            "right": torch.tensor(right, device=dev),
            "cls": torch.tensor(cls, device=dev)}


@torch.no_grad()
def predict(tree: dict, X: torch.Tensor) -> torch.Tensor:
    """Route every row of X (B, I) bits to its leaf; return (B,) predicted classes."""
    B = X.shape[0]
    dev = X.device
    feat, left, right, cls = tree["feat"], tree["left"], tree["right"], tree["cls"]
    node = torch.zeros(B, dtype=torch.long, device=dev)
    ar = torch.arange(B, device=dev)
    for _ in range(int(feat.numel())):                   # <= number of nodes; leaves self-loop
        f = feat[node]
        leaf = f < 0
        if bool(leaf.all()):
            break
        bit = X[ar, f.clamp_min(0)] > 0
        child = torch.where(bit, right[node], left[node])
        node = torch.where(leaf, node, child)
    return cls[node]


@torch.no_grad()
def ensemble_scores(trees: list, alphas: list, X: torch.Tensor) -> torch.Tensor:
    """(B, CLS) confidence-weighted class votes of a boosted forest over rows of X."""
    s = torch.zeros(X.shape[0], CLS, device=X.device)
    ar = torch.arange(X.shape[0], device=X.device)
    for tr, a in zip(trees, alphas):
        s[ar, predict(tr, X)] += a
    return s


# ==========================================================================================
# Boolean extraction: each tree -> list of (conjunction, class). The ensemble is those DNFs
# voting with weights alpha (a threshold-of-DNFs circuit).
# ==========================================================================================
def export_rules(tree: dict) -> list[tuple[list[tuple[int, int]], int]]:
    """Enumerate root->leaf paths. Each path is a conjunction of (global_bit, value in {0,1})
    literals; value 1 means the bit is set. Returns (literals, leaf_class) per leaf."""
    feat = tree["feat"].tolist()
    left = tree["left"].tolist()
    right = tree["right"].tolist()
    cls = tree["cls"].tolist()
    rules: list[tuple[list[tuple[int, int]], int]] = []

    def walk(nid: int, lits: list[tuple[int, int]]) -> None:
        if feat[nid] < 0:
            rules.append((list(lits), cls[nid]))
            return
        f = feat[nid]
        walk(left[nid], lits + [(f, 0)])
        walk(right[nid], lits + [(f, 1)])

    walk(0, [])
    return rules


# ==========================================================================================
# The SAMME boosting loop, factored out so conv_boost.py reuses the SAME head fit.
# ==========================================================================================
def fit_boost(X: torch.Tensor, y: torch.Tensor, *, n_trees: int, max_leaves: int, min_leaf: int,
              max_features: int, lr: float, max_depth: int, seed: int,
              evalsets: dict | None = None, logpath: Path | None = None,
              tag: str = "") -> tuple[list, list]:
    """Multiclass AdaBoost (SAMME) over deep single-bit trees. Build on weighted data, up-weight
    every misclassification, alpha = (ln((1-e)/e) + ln(K-1)) * lr, reset weights on a perfect
    tree, drop a worse-than-random tree. If evalsets is given, log the running confidence-weighted
    ensemble accuracy per tree (the boosting learning curve). Returns (trees, alphas) on CPU."""
    dev = X.device
    N, I = X.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    yoh = torch.zeros(N, CLS, device=dev)
    yoh[torch.arange(N, device=dev), y] = 1.0
    w = torch.full((N,), 1.0 / N, device=dev)            # sample weights (start uniform=balanced)
    MF = max_features or I
    trees: list = []
    alphas: list = []
    score = {n: torch.zeros(len(yy), CLS, device=dev) for n, (_, yy) in (evalsets or {}).items()}
    t0 = time.time()
    for t in range(n_trees):
        gfeat = (torch.randperm(I, generator=g)[:MF].sort().values.to(dev)
                 if MF < I else torch.arange(I, device=dev))
        tree = build_tree(X[:, gfeat], gfeat, y, w, yoh * w[:, None], max_leaves, min_leaf,
                          max_depth)
        miss = (predict(tree, X) != y).to(torch.float32)
        err = float((w * miss).sum() / w.sum())

        if err >= 1.0 - 1.0 / CLS:                       # worse than random: drop, fresh subspace
            w = torch.full((N,), 1.0 / N, device=dev)
            continue
        if err <= 1e-12:                                 # perfect on weighted train
            alpha = (math.log((1 - 1e-12) / 1e-12) + math.log(CLS - 1)) * lr
            w = torch.full((N,), 1.0 / N, device=dev)    # reset so the next tree explores anew
        else:
            alpha = (math.log((1 - err) / err) + math.log(CLS - 1)) * lr
            w = w * torch.exp(alpha * miss)              # up-weight EVERY misclassification
            w = w / w.sum()

        trees.append({k: v.cpu() for k, v in tree.items()})   # store off-GPU (votes live in score)
        alphas.append(alpha)

        if evalsets:                                     # accumulate this tree's vote, then eval
            rec = {"tree": t, "err": round(err, 5), "alpha": round(alpha, 4),
                   "min": round((time.time() - t0) / 60, 2)}
            for n, (Xe, ye) in evalsets.items():
                s = score[n]
                s[torch.arange(s.shape[0], device=dev), predict(tree, Xe)] += alpha
                rec[n] = round(100.0 * float((s.argmax(1) == ye).float().mean()), 2)
            if logpath:
                with open(logpath, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            print(f"{tag}tree {t:4d} | err {err:.4f} a {alpha:5.2f} | "
                  + " ".join(f"{n} {rec[n]:5.2f}" for n in evalsets) + f" | {rec['min']:5.2f}m",
                  flush=True)
    return trees, alphas


# ==========================================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/cifar-10-batches-py"))
    p.add_argument("--download", action="store_true")
    p.add_argument("--num-bits", type=int, default=4, help="thermometer bits per channel")
    p.add_argument("--max-leaves", type=int, default=512,
                   help="leaf budget per tree (leaf-wise best-first) = max DNF terms; small "
                        "budgets = weak learners = the healthy boosting regime")
    p.add_argument("--depth", type=int, default=32, help="max tree depth (safety guard only)")
    p.add_argument("--trees", type=int, default=1500, help="boosting rounds")
    p.add_argument("--lr", type=float, default=0.3,
                   help="shrinkage: scale each tree's vote by lr (<1 needs more trees, generalises "
                        "better -- standard boosting shrinkage)")
    p.add_argument("--min-leaf", type=int, default=4, help="min RAW samples per leaf")
    p.add_argument("--max-features", type=int, default=2048,
                   help="feature bits sampled per tree (diversity + speed); 0 = all")
    p.add_argument("--hflip", action="store_true", help="augment train with horizontal flips")
    p.add_argument("--aug-crops", type=int, default=0,
                   help="augment train with N per-image random reflect-pad crops (extra rows)")
    p.add_argument("--aug-pad", type=int, default=4, help="crop reflect-pad radius (px)")
    p.add_argument("--out", type=Path, required=True, help="prefix for .jsonl and .pkl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    dev = args.device

    # ---- data + lean binary features -----------------------------------------------------
    tx, ty, ex, ey = load_cifar10(args.data_dir, args.download)
    nv = 5000
    sample = tx[:2000]
    enc32 = Thermometer(num_bits=args.num_bits).fit(sample).to(dev)

    @torch.no_grad()
    def encode(images: torch.Tensor) -> torch.Tensor:      # chunked to bound peak memory
        outs = [build_features(images[i:i + 4096].to(dev), enc32)
                for i in range(0, len(images), 4096)]
        return torch.cat(outs, 0)

    trx, trY = tx[:-nv], ty[:-nv]
    if args.hflip or args.aug_crops:
        trx, trY = augment_train(trx, trY, hflip=args.hflip, crops=args.aug_crops,
                                 pad=args.aug_pad, seed=args.seed)
    Xtr, ytr = encode(trx), trY.to(dev)
    Xva, yva = encode(tx[-nv:]), ty[-nv:].to(dev)
    Xte, yte = encode(ex), ey.to(dev)
    N, I = Xtr.shape
    MF = args.max_features or I

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".jsonl").write_text("")
    print(f"boost I={I} leaves={args.max_leaves} depth<={args.depth} trees={args.trees} "
          f"lr={args.lr} min_leaf={args.min_leaf} max_features={MF} "
          f"aug(hflip={args.hflip},crops={args.aug_crops}) train={N} val={nv} "
          f"device={dev}", flush=True)

    # ---- SAMME boosting ------------------------------------------------------------------
    evalsets = {"train": (Xtr, ytr), "val": (Xva, yva), "test": (Xte, yte)}
    trees, alphas = fit_boost(Xtr, ytr, n_trees=args.trees, max_leaves=args.max_leaves,
                              min_leaf=args.min_leaf, max_features=MF, lr=args.lr,
                              max_depth=args.depth, seed=args.seed, evalsets=evalsets,
                              logpath=out.with_suffix(".jsonl"))

    # ---- save ensemble (+ encoder thresholds) for Boolean extraction ---------------------
    with open(out.with_suffix(".pkl"), "wb") as f:
        pickle.dump({"args": vars(args) | {"out": str(out)},
                     "alphas": alphas, "trees": trees,      # trees already on CPU
                     "thr32": enc32.thresholds.cpu(),
                     "offsets": OFFS, "num_bits": args.num_bits, "I": I}, f)
    print(f"saved {len(trees)} trees -> {out.with_suffix('.pkl')}", flush=True)


if __name__ == "__main__":
    main()
