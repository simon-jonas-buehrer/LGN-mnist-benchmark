"""tree_scratch: a BOOSTED DECISION-TREE ensemble over the thermometer bits.

The third scratch line (after scratch/opt.py's monarch LUT net and crazy_scratch's free-NAND
DAG). Same binary substrate -- the Thermometer encoder's I input bits -- but here the model is
a forest of DEEP single-bit decision trees combined by CONFIDENCE-WEIGHTED VOTING, built one
tree at a time on the CURRENT MISTAKES. This is multiclass AdaBoost (SAMME); it is exactly the
mechanism worked out in the design thread:

  * SINGLE-BIT SPLITS. Each internal node tests one thermometer bit `x[f]` (is pixel-channel c
    above threshold t?). A root->leaf path is therefore a CONJUNCTION of bit literals, and a
    tree is a DNF over the bits -- the logic is a mechanical read-off (see export_rules). We do
    NOT use monomial/parity split features: a depth-d tree already expresses any d-way
    interaction (XOR included) with plain single-bit splits; depth is the knob, not the feature
    family. The only cost is greed being blind to an interaction until it is inside it -- watch
    for trees stalling at weighted error ~1-1/K, the parity-stall signature (harmless on CIFAR).

  * DEEP. --depth is unbounded in spirit (default 20). Deep trees are strong learners, which
    would make classic AdaBoost degenerate (a tree hitting weighted error 0 => alpha->inf, the
    ensemble stalls on one tree). Two things keep "many trees on mistakes" alive: per-tree
    FEATURE SUBSAMPLING (--max-features) makes trees diverse and keeps their weighted error > 0,
    and a perfect tree resets the weights so the next tree explores a different subspace.

  * VOTE, not OR. OR-of-trees is monotone -- it can only fix false negatives, never false
    positives. A weighted vote can push either way, so it fixes BOTH. In multiclass every
    misclassification (whatever the confusion) is "a wrong prediction" and is up-weighted
    symmetrically -- the FP-and-FN-in-one-buffer idea, generalised to K classes.

  * WEIGHTS = the mistake buffer. We do NOT train the next tree on the errors alone (that
    oscillates -- tree t+1 forgets why the correct ones were correct). We keep ALL data and
    REWEIGHT: misclassified samples up by exp(alpha), correct ones implicitly down after
    renorm, so the next tree's Gini concentrates on the errors without abandoning the rest.

  * VOTE WEIGHT = competence. alpha_t = ln((1-e_t)/e_t) + ln(K-1) (SAMME). A coin-flip tree
    (e=1-1/K) gets alpha=0 and is silenced automatically; a good tree gets a big vote.

  * READOUT. Prediction is argmax_c sum_t alpha_t * 1[tree_t(x)=c] -- a weighted-majority /
    threshold circuit over the trees. Boolean, hardware-friendly (popcount + compare), but a
    threshold-of-DNFs, NOT a single flat minimal DNF -- the price of fixing both error types.

Self-contained (only torch + the repo's encoder/data), device-agnostic: runs on CPU here and
on CUDA automatically when a GPU is present. No augmentation -- boosting maintains per-sample
weights over a FIXED training set, which augmentation would break.

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


def build_features(imgs: torch.Tensor, enc32: Thermometer, enc16: Thermometer) -> torch.Tensor:
    """Spatial binary feature map for a batch of images -> (B, F) uint8 bits. All bits, so the
    ensemble stays Boolean-extractable. The lever over flat thermometer bits is LOCALITY:
    edge-sign (census) bits + OR-pooled edges give the trees the spatial structure that
    axis-aligned splits on independent pixel bits cannot see."""
    cen = census(imgs, OFFS, 2)                            # (B, C*8, 32, 32) edge signs
    r, g, b = imgs[:, 0:1], imgs[:, 1:2], imgs[:, 2:3]
    feats = [
        enc32(imgs).flatten(1),                            # thermometer intensity @ 32x32
        cen.flatten(1),                                    # edge signs @ 32x32  (main lever)
        torch.cat([r > g, g > b, r > b], 1).flatten(1),    # color-order bits
        enc16(F.avg_pool2d(imgs, 2)).flatten(1),           # thermometer context @ 16x16
        F.max_pool2d(cen.float(), 2).flatten(1),           # OR-pooled edges @ 16x16 (transl.)
    ]
    return torch.cat([f.to(torch.uint8) for f in feats], 1)


# ==========================================================================================
# One deep single-bit decision tree, built greedily on weighted samples (Gini).
#
# A tree is four parallel arrays over its nodes: feat (GLOBAL bit index, -1 at a leaf), left,
# right (child node ids), cls (the leaf's weighted-majority class). Splits are chosen among a
# per-tree random subset of features for speed + diversity, but feat stores the GLOBAL id so
# prediction and extraction need only the full bit matrix.
# ==========================================================================================
def build_tree(Xsub: torch.Tensor, gfeat: torch.Tensor, y: torch.Tensor, w: torch.Tensor,
               wyoh: torch.Tensor, max_leaves: int, min_leaf: int, max_depth: int) -> dict:
    """LEAF-WISE (best-first) tree: repeatedly split the leaf whose split most reduces impurity,
    WHEREVER it sits, until a leaf BUDGET is spent -- not a balanced depth-d tree. Capacity lands
    only where the data needs a decision, so trees are unbalanced and sparse; #leaves = #DNF
    terms is bounded by max_leaves, which keeps the extracted Boolean logic small. max_depth is
    just a safety guard.

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


# ==========================================================================================
# Boolean extraction: each tree -> list of (conjunction, class); the ensemble is those DNFs
# voting with weights alpha. Saved so the logic can be read/minimised later.
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
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/cifar-10-batches-py"))
    p.add_argument("--download", action="store_true")
    p.add_argument("--num-bits", type=int, default=4, help="thermometer bits per channel")
    p.add_argument("--max-leaves", type=int, default=256,
                   help="leaf budget per tree (leaf-wise best-first growth) = max DNF terms; "
                        "the primary capacity knob, keeps extracted logic sparse")
    p.add_argument("--depth", type=int, default=32, help="max tree depth (safety guard only)")
    p.add_argument("--trees", type=int, default=200, help="boosting rounds")
    p.add_argument("--lr", type=float, default=1.0,
                   help="shrinkage: scale each tree's vote weight by lr (<1 needs more trees but "
                        "generalises better -- standard boosting shrinkage)")
    p.add_argument("--min-leaf", type=int, default=5, help="min RAW samples per leaf")
    p.add_argument("--max-features", type=int, default=1024,
                   help="features sampled per tree (diversity + speed); 0 = all")
    p.add_argument("--out", type=Path, required=True, help="prefix for .jsonl and .pkl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    dev = args.device
    g = torch.Generator(device="cpu").manual_seed(args.seed)

    # ---- data + SPATIAL binary features --------------------------------------------------
    tx, ty, ex, ey = load_cifar10(args.data_dir, args.download)
    nv = 5000
    sample = tx[:2000]
    enc32 = Thermometer(num_bits=args.num_bits).fit(sample).to(dev)
    enc16 = Thermometer(num_bits=args.num_bits).fit(F.avg_pool2d(sample, 2)).to(dev)

    @torch.no_grad()
    def encode(images: torch.Tensor) -> torch.Tensor:      # chunked to bound peak memory
        outs = [build_features(images[i:i + 4096].to(dev), enc32, enc16)
                for i in range(0, len(images), 4096)]
        return torch.cat(outs, 0)

    Xtr, ytr = encode(tx[:-nv]), ty[:-nv].to(dev)
    Xva, yva = encode(tx[-nv:]), ty[-nv:].to(dev)
    Xte, yte = encode(ex), ey.to(dev)
    N, I = Xtr.shape
    MF = args.max_features or I                          # features sampled per tree
    yoh = torch.zeros(N, CLS, device=dev)                # one-hot targets, reused every round
    yoh[torch.arange(N, device=dev), ytr] = 1.0
    # No persistent float copy of the full (N, I) matrix -- only the per-tree subsampled
    # columns are cast to float (below), which keeps the GPU footprint small enough to
    # co-locate runs on one card.

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".jsonl").write_text("")
    print(f"boost I={I} leaves={args.max_leaves} depth<={args.depth} trees={args.trees} "
          f"min_leaf={args.min_leaf} max_features={MF} train={N} val={nv} device={dev}", flush=True)

    # ---- SAMME boosting loop -------------------------------------------------------------
    w = torch.full((N,), 1.0 / N, device=dev)            # sample weights (start uniform=balanced)
    trees: list[dict] = []
    alphas: list[float] = []
    score = {"train": torch.zeros(N, CLS, device=dev),   # running weighted votes per set
             "val": torch.zeros(nv, CLS, device=dev),
             "test": torch.zeros(len(yte), CLS, device=dev)}
    sets = {"train": (Xtr, ytr), "val": (Xva, yva), "test": (Xte, yte)}
    t0 = time.time()

    for t in range(args.trees):
        # per-tree feature subsample (global ids), then build on that subspace
        gfeat = (torch.randperm(I, generator=g)[:MF].sort().values.to(dev)
                 if MF < I else torch.arange(I, device=dev))
        wyoh = yoh * w[:, None]
        Xsub = Xtr[:, gfeat]                             # (N, F) uint8; cast per-node inside build
        tree = build_tree(Xsub, gfeat, ytr, w, wyoh, args.max_leaves, args.min_leaf, args.depth)

        pred_tr = predict(tree, Xtr)
        miss = (pred_tr != ytr).to(torch.float32)
        err = float((w * miss).sum() / w.sum())

        if err >= 1.0 - 1.0 / CLS:                       # worse than random guessing: drop it
            print(f"tree {t:3d}: err {err:.4f} >= {1 - 1/CLS:.4f} (random) -- skipped", flush=True)
            w = torch.full((N,), 1.0 / N, device=dev)    # reset and try a fresh subspace
            continue

        if err <= 1e-12:                                 # perfect on weighted train
            alpha = (math.log((1 - 1e-12) / 1e-12) + math.log(CLS - 1)) * args.lr
            w = torch.full((N,), 1.0 / N, device=dev)    # reset so the next tree explores anew
        else:
            alpha = (math.log((1 - err) / err) + math.log(CLS - 1)) * args.lr
            w = w * torch.exp(alpha * miss)              # up-weight EVERY misclassification
            w = w / w.sum()

        trees.append({k: v.cpu() for k, v in tree.items()})  # store off-GPU: past trees are
        alphas.append(alpha)                                  # never re-run (votes are in score)

        # accumulate this tree's weighted vote into every set's running score, then eval
        rec = {"tree": t, "err": round(err, 5), "alpha": round(alpha, 4),
               "min": round((time.time() - t0) / 60, 2)}
        for name, (X, yv) in sets.items():
            s = score[name]
            s[torch.arange(s.shape[0], device=dev), predict(tree, X)] += alpha
            rec[name] = round(100.0 * float((s.argmax(1) == yv).float().mean()), 2)
        with open(out.with_suffix(".jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"tree {t:3d} | err {err:.4f} a {alpha:5.2f} | "
              f"train {rec['train']:5.2f} val {rec['val']:5.2f} test {rec['test']:5.2f} | "
              f"{rec['min']:5.2f}m", flush=True)

    # ---- save ensemble (+ encoder thresholds) for Boolean extraction ---------------------
    with open(out.with_suffix(".pkl"), "wb") as f:
        pickle.dump({"args": vars(args) | {"out": str(out)},
                     "alphas": alphas, "trees": trees,      # trees already on CPU
                     "thr32": enc32.thresholds.cpu(), "thr16": enc16.thresholds.cpu(),
                     "offsets": OFFS, "num_bits": args.num_bits, "I": I}, f)
    print(f"saved {len(trees)} trees -> {out.with_suffix('.pkl')}", flush=True)


if __name__ == "__main__":
    main()
