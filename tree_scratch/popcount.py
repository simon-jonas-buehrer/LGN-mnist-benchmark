"""tree_scratch (simplified): a POPCOUNT FOREST over the thermometer bits.

The simplest possible tree ensemble on the Boolean substrate. We drop everything clever from
boost.py -- no SAMME reweighting, no confidence weights alpha, no OR-pooling, no convolution.
Just:

  * MANY INDEPENDENT TREES. Grow `--trees` deep single-bit decision trees, each on its own
    random BAG of rows (bootstrap-style subsample) and its own random SUBSET of feature bits.
    That is plain bagging + feature subsampling = a random forest. Trees never see each other;
    the whole thing is embarrassingly parallel, unlike boosting's sequential mistake-chasing.

  * EACH TREE IS A MULTICLASS VOTER. A root->leaf path is a CONJUNCTION of bit literals and the
    leaf carries its majority class, so a tree is a set of Boolean rules (a DNF per class) that
    emits exactly ONE class per input -- a one-hot vote.

  * READOUT = POPCOUNT. Stack the one-hot votes of all trees and COUNT how many voted for each
    class: score_c(x) = #{ t : tree_t(x) = c }. Predict argmax_c score_c. No weights -- every
    tree counts as 1. This is the "popcount over all trees tells us the class" idea: on hardware
    it is route-to-leaf then popcount a bit column per class, then compare. It is exactly the
    readout of a Tsetlin machine / difflogic net (sum of clause votes per class), reached here
    from decision trees instead of learned gates.

Why a vote and not boosting? Boosting fixes both error types by chasing mistakes with weighted
votes -- powerful but sequential and finicky (see boost.py's parity-stall / perfect-tree resets).
A plain vote leans on the OTHER classic mechanism: many DIVERSE weak-ish trees whose independent
errors cancel under the count (Breiman's random forests; the margin/voting view of ensembles).
No per-sample weights to maintain, so it stays dead simple and Boolean-extractable.

Literature (this is not a new idea -- we've re-derived a known readout from trees):
  * Tsetlin machine (Granmo 2018) -- per-class SUM of conjunctive-clause votes, argmax across
    classes. Our per-class leaf-vote popcount is exactly that; clauses <-> our root->leaf ANDs.
    Their polarity trick (NEGATIVE-voting clauses per class) sharpens margins -- a natural
    extension of pure popcount to a SIGNED vote.
  * Differentiable Logic Gate Networks (Petersen 2022) end in `GroupSum`: partition the final
    gates into class groups and popcount per group -> argmax. Identical aggregation, over gates.
  * Random forests (Breiman 2001): plurality vote of decorrelated trees; error ~ rho_bar(1-s^2)/s^2
    -- DIVERSITY (low correlation) matters more than per-tree strength, so bag rows + subsample
    bits and keep trees shallow-ish. Margin theory (Schapire et al. 1998) says test error keeps
    dropping as the vote-gap distribution thickens, even after train acc saturates -> add trees.
  * Ceiling: axis-aligned single-bit forests on binarized pixels hit ~96-97% MNIST but only
    ~50-60% CIFAR-10 without spatial structure; Conv-Tsetlin ~75%, Conv-DiffLogic 86.3% needed
    locality + OR-pooling + residual init. So this lean version is a clean BASELINE, not a peak.

Self-contained (torch + the repo's encoder/data), device-agnostic (CPU here, CUDA when present).

    python tree_scratch/popcount.py --out tree_scratch/runs/pc0
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


def build_features(imgs: torch.Tensor, enc32: Thermometer) -> torch.Tensor:
    """Lean binary feature map -> (B, F) uint8 bits, all Boolean so the forest stays extractable.
    Deliberately NO pooling and NO convolution (that was the point of the simplification): just
    thermometer intensity bits, per-pixel census edge-sign bits (cheap locality), and color-order
    bits. Add spatial context back via --features spatial only if a run needs it."""
    cen = census(imgs, OFFS, 2)                            # (B, C*8, 32, 32) edge signs
    r, g, b = imgs[:, 0:1], imgs[:, 1:2], imgs[:, 2:3]
    feats = [
        enc32(imgs).flatten(1),                            # thermometer intensity @ 32x32
        cen.flatten(1),                                    # edge signs @ 32x32  (locality)
        torch.cat([r > g, g > b, r > b], 1).flatten(1),    # color-order bits
    ]
    return torch.cat([f.to(torch.uint8) for f in feats], 1)


# ==========================================================================================
# One deep single-bit decision tree, grown greedily (leaf-wise, Gini) on a bag of rows.
# Four parallel arrays over nodes: feat (GLOBAL bit index, -1 at a leaf), left, right (child
# ids), cls (leaf majority class). Splits are chosen on the tree's feature subset; feat stores
# the GLOBAL id so prediction/extraction need only the full bit matrix.
# ==========================================================================================
def build_tree(Xsub: torch.Tensor, gfeat: torch.Tensor, y: torch.Tensor,
               max_leaves: int, min_leaf: int, max_depth: int) -> dict:
    """LEAF-WISE (best-first): repeatedly split the leaf whose split most cuts Gini impurity,
    until the leaf budget is spent -- capacity lands only where the data needs a decision, so
    #leaves = #DNF terms is bounded by max_leaves. max_depth is a safety guard.

    Xsub (N, F) uint8 bits on the tree's subsampled features; gfeat (F,) their global ids;
    y (N,) long labels of the bag rows (uniform weight -- plain forest, no boosting weights)."""
    dev = Xsub.device
    N = Xsub.shape[0]
    yoh = torch.zeros(N, CLS, device=dev)
    yoh[torch.arange(N, device=dev), y] = 1.0             # one-hot; counts are just sums of these
    feat: list[int] = []
    left: list[int] = []
    right: list[int] = []
    cls: list[int] = []

    def add_leaf(idx: torch.Tensor):                     # append a leaf, return (node_id, cw)
        cw = yoh[idx].sum(0)                             # (K,) class counts in this leaf
        nid = len(feat)
        feat.append(-1)
        left.append(-1)
        right.append(-1)
        cls.append(int(cw.argmax()))
        return nid, cw

    def best_split(idx: torch.Tensor, cw: torch.Tensor):  # -> (gain, gfeat, lidx, ridx) | None
        Xg = Xsub[idx].to(torch.float32)                # (n, F); cast only this subset (memory)
        b1 = yoh[idx].t() @ Xg                           # (K, F) count of bit==1 per class
        raw1 = Xg.sum(0)                                 # (F,)   raw count of bit==1
        n = idx.numel()
        cntR = b1                                        # bit==1 goes right
        cntL = cw[:, None] - b1                          # bit==0 goes left
        valid = (raw1 >= min_leaf) & (n - raw1 >= min_leaf)
        # Gini surrogate: maximise sum_c cnt^2 / n_side (= minimise Gini impurity).
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

    root_idx = torch.arange(N, device=dev)
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
# Boolean extraction: each tree -> list of (conjunction, class). The whole forest is those
# rules voting 1-each; the class with the most firing rules wins (popcount).
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
    p.add_argument("--max-leaves", type=int, default=512,
                   help="leaf budget per tree (leaf-wise best-first) = max DNF terms/tree")
    p.add_argument("--depth", type=int, default=32, help="max tree depth (safety guard only)")
    p.add_argument("--trees", type=int, default=200, help="number of trees in the forest")
    p.add_argument("--min-leaf", type=int, default=5, help="min RAW samples per leaf")
    p.add_argument("--max-features", type=int, default=1024,
                   help="feature bits sampled per tree (RF diversity + speed); 0 = all")
    p.add_argument("--bag-frac", type=float, default=0.7,
                   help="fraction of train rows in each tree's bag (row subsample, no replace)")
    p.add_argument("--out", type=Path, required=True, help="prefix for .jsonl and .pkl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    dev = args.device
    g = torch.Generator(device="cpu").manual_seed(args.seed)

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

    Xtr, ytr = encode(tx[:-nv]), ty[:-nv].to(dev)
    Xva, yva = encode(tx[-nv:]), ty[-nv:].to(dev)
    Xte, yte = encode(ex), ey.to(dev)
    N, I = Xtr.shape
    MF = args.max_features or I                          # feature bits sampled per tree
    bag_n = max(2 * args.min_leaf, int(round(args.bag_frac * N)))

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".jsonl").write_text("")
    print(f"popcount-forest I={I} leaves={args.max_leaves} depth<={args.depth} "
          f"trees={args.trees} min_leaf={args.min_leaf} max_features={MF} "
          f"bag={bag_n}/{N} val={nv} device={dev}", flush=True)

    # ---- forest loop: independent trees, votes summed (popcount) --------------------------
    trees: list[dict] = []
    vote = {"train": torch.zeros(N, CLS, device=dev),    # running vote COUNT per class per set
            "val": torch.zeros(nv, CLS, device=dev),
            "test": torch.zeros(len(yte), CLS, device=dev)}
    sets = {"train": (Xtr, ytr), "val": (Xva, yva), "test": (Xte, yte)}
    t0 = time.time()

    for t in range(args.trees):
        bag = torch.randperm(N, generator=g)[:bag_n].to(dev)          # row subsample (bagging)
        gfeat = (torch.randperm(I, generator=g)[:MF].sort().values.to(dev)
                 if MF < I else torch.arange(I, device=dev))          # feature subsample
        Xb = Xtr[bag][:, gfeat]                                       # (bag_n, F) uint8
        tree = build_tree(Xb, gfeat, ytr[bag], args.max_leaves, args.min_leaf, args.depth)
        trees.append({k: v.cpu() for k, v in tree.items()})           # store off-GPU

        # cast one vote per tree into every set's running count, then eval the popcount argmax
        rec = {"tree": t, "min": round((time.time() - t0) / 60, 2)}
        for name, (X, yv) in sets.items():
            s = vote[name]
            s[torch.arange(s.shape[0], device=dev), predict(tree, X)] += 1.0   # popcount
            rec[name] = round(100.0 * float((s.argmax(1) == yv).float().mean()), 2)
        with open(out.with_suffix(".jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"tree {t:3d} | train {rec['train']:5.2f} val {rec['val']:5.2f} "
              f"test {rec['test']:5.2f} | {rec['min']:5.2f}m", flush=True)

    # ---- save forest (+ encoder thresholds) for Boolean extraction ------------------------
    with open(out.with_suffix(".pkl"), "wb") as f:
        pickle.dump({"args": vars(args) | {"out": str(out)},
                     "trees": trees,                        # already on CPU; each votes 1
                     "thr32": enc32.thresholds.cpu(),
                     "offsets": OFFS, "num_bits": args.num_bits, "I": I}, f)
    print(f"saved {len(trees)} trees -> {out.with_suffix('.pkl')}", flush=True)


if __name__ == "__main__":
    main()
