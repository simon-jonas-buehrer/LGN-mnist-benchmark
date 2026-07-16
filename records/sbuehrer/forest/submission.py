"""sbuehrer/forest: a boosted decision-tree forest, emitted as combinational logic.

A decision tree over binary features IS a boolean function. Each root-to-leaf path is a
conjunction of literals, so a tree's leaf indicators are a shared-prefix AND network and the set
of paths reaching a class is a DNF. Boost a forest of them, weight each tree by an integer, sum
per class and argmax, and the whole model is a circuit -- no gradient, no LUT net, no search.

The optimizer is SAMME (multiclass AdaBoost): build a tree on the CURRENT MISTAKES, weight it by
its confidence, up-weight everything it got wrong, repeat. The tree builder is leaf-wise
best-first on weighted Gini, and over binary features a split search is a single GEMM
(`wyoh.t() @ X` counts, per class, how many weighted samples have each bit set), so the whole
forest trains in seconds.

Three things make this cheap in silicon:

  Leaf indicators as `reach` wires, not flat ANDs. reach(child) = reach(parent) & +/-literal is
  exactly 2 gates per internal node, so a tree costs 2(L-1) gates REGARDLESS OF ITS SHAPE. Area
  depends on the leaf count and not on the depth, which is why max_depth here is non-binding:
  constraining depth would only remove capacity at zero area saving. Emitting each leaf as a flat
  AND of its d literals and hoping ABC rediscovers the sharing would hand it a 4-5x larger AIG, so
  we give ABC the intra-tree sharing we know exactly and let it find the inter-tree sharing we
  don't.

  The class indicator partitions. Each leaf carries exactly ONE class, so the ten per-class ORs
  are disjoint and cost ~L per tree in total, and a single reach network scores all ten classes.

  Bit-plane popcount. w_t is a constant and v[t][c] is one bit, so
  score_c = sum_b 2^b * popcount({v[t][c] : bit b of w_t set}). A zero weight-bit contributes no
  hardware at all, which halves the adder inputs versus a Wallace tree over T B-bit numbers.

The encoder is the harness's own thermometer (hw.even_thresholds), so this record's `bits` means
exactly what it means in the backprop and genetic records and the comparison is at matched input
encoding. Thermometer bits rather than the raw pixel bits, even though the raw bits are free wires
and strictly more expressive: greedy Gini cannot use a bit like pix[6], which is non-monotone in
intensity and has ~zero standalone information gain, so it is never selected. Thermometer bits are
monotone and therefore individually informative, which is what a greedy builder needs. What the
sweeps said about where to spend the budget (encoder resolution, and the tree/leaf split at
matched silicon) is in the README.

Weights are quantized INSIDE the boosting loop, not rounded afterwards. The sample-weight update
consumes the integer alpha the circuit will use, so every later tree is fit against the residual
error of the circuit-exact ensemble and the quantization error is boosted away instead of
accumulating. `wscale` is auto-derived per run from the observed alpha distribution (a first
unquantized pass sets it from p95), never hand-tuned. SAMME drops any tree with err >= 1-1/K, so
alpha >= 0 always: every score is unsigned and there is no two's complement anywhere.

The tree builder is tree_scratch/boost.py @ git 4111db6, ported to MNIST.
"""

from __future__ import annotations

import heapq
import math

import numpy as np
import torch

from mnistbench.data import Mnist, N_CLASSES, N_PIXELS, PIXEL_BITS
from mnistbench.hw import even_thresholds
from mnistbench.spec import Submission

TITLE = "forest (SAMME-boosted decision trees, integer-weighted vote)"

# `bits` thermometer bits per pixel (hw.even_thresholds, same as the other records), `leaves` the
# per-tree leaf budget (the ONLY capacity knob -- depth is free, see the docstring), `wbits` the
# integer tree-weight width. `trees` is the boosting round count.
#
# Not hand-picked: every point is the measured-Pareto winner for its area. A 72-config grid over
# leaves x wbits x bits was swept (the tree-count axis comes free, since the first t trees ARE the
# round-t ensemble), 39 candidates were then SYNTHESIZED, and these 7 are what survived on real
# silicon, thinned to ~1.8x steps. Selection is on val; the harness reports test from the netlist.
POINTS = [
    {"name": "xxs", "trees": 5, "leaves": 8, "wbits": 2, "bits": 3},
    {"name": "xs", "trees": 9, "leaves": 16, "wbits": 2, "bits": 7},
    {"name": "s", "trees": 5, "leaves": 128, "wbits": 3, "bits": 3},
    {"name": "m", "trees": 13, "leaves": 128, "wbits": 3, "bits": 3},
    {"name": "l", "trees": 30, "leaves": 128, "wbits": 2, "bits": 7},
    {"name": "xl", "trees": 39, "leaves": 256, "wbits": 2, "bits": 3},
    {"name": "xxl", "trees": 78, "leaves": 256, "wbits": 2, "bits": 7},
]

NEG_INF = float("-inf")


# ==========================================================================================
# The tree builder, ported from tree_scratch/boost.py @ 4111db6. Four parallel arrays over
# nodes: feat (GLOBAL bit index, -1 at a leaf), left, right (child ids), cls (leaf weighted-
# majority class).
# ==========================================================================================
def build_tree(X: torch.Tensor, y: torch.Tensor, w: torch.Tensor, wyoh: torch.Tensor,
               max_leaves: int, min_leaf: int) -> dict:
    """LEAF-WISE (best-first): repeatedly split the leaf whose split most cuts weighted Gini,
    until the leaf budget is spent. No depth cap: area is 2(L-1) whatever the shape."""
    dev = X.device
    K = wyoh.shape[1]
    feat: list[int] = []
    left: list[int] = []
    right: list[int] = []
    cls: list[int] = []

    def add_leaf(idx: torch.Tensor):
        cw = torch.zeros(K, device=dev).index_add_(0, y[idx], w[idx])
        nid = len(feat)
        feat.append(-1)
        left.append(-1)
        right.append(-1)
        cls.append(int(cw.argmax()))
        return nid, cw

    def best_split(idx: torch.Tensor, cw: torch.Tensor):
        Xg = X[idx].to(torch.float32)
        b1 = wyoh[idx].t() @ Xg                          # (K, F) weighted count of bit==1
        raw1 = Xg.sum(0)
        n = idx.numel()
        cntR = b1                                        # bit==1 goes right
        cntL = cw[:, None] - b1                          # bit==0 goes left
        valid = (raw1 >= min_leaf) & (n - raw1 >= min_leaf)
        score = (cntL.pow(2).sum(0) / cntL.sum(0).clamp_min(1e-12)
                 + cntR.pow(2).sum(0) / cntR.sum(0).clamp_min(1e-12))
        score = torch.where(valid, score, torch.full_like(score, NEG_INF))
        base = cw.pow(2).sum() / cw.sum().clamp_min(1e-12)
        best = int(score.argmax())
        gain = float(score[best]) - float(base)
        if not bool(valid[best]) or gain <= 1e-9:
            return None
        bit = Xg[:, best] > 0
        return gain, best, idx[~bit], idx[bit]

    heap: list = []
    ctr = 0

    def consider(nid: int, idx: torch.Tensor, cw: torch.Tensor):
        nonlocal ctr
        if idx.numel() < 2 * min_leaf:
            return
        bs = best_split(idx, cw)
        if bs is not None:
            heapq.heappush(heap, (-bs[0], ctr, nid, idx, bs))
            ctr += 1

    root_idx = torch.arange(X.shape[0], device=dev)
    root, cw0 = add_leaf(root_idx)
    consider(root, root_idx, cw0)
    leaves = 1
    while heap and leaves < max_leaves:
        _, _, nid, idx, bs = heapq.heappop(heap)
        _, gf, lidx, ridx = bs
        lc, lcw = add_leaf(lidx)
        rc, rcw = add_leaf(ridx)
        feat[nid], left[nid], right[nid] = gf, lc, rc     # the popped leaf becomes internal
        leaves += 1
        consider(lc, lidx, lcw)
        consider(rc, ridx, rcw)

    return {"feat": np.array(feat, np.int64), "left": np.array(left, np.int64),
            "right": np.array(right, np.int64), "cls": np.array(cls, np.int64)}


def _route(tree: dict, X) -> "np.ndarray | torch.Tensor":
    """Route every row to its leaf; return the leaf node id per row. Works for numpy or torch."""
    is_t = isinstance(X, torch.Tensor)
    n = X.shape[0]
    feat = tree["feat"]
    if is_t:
        dev = X.device
        feat_t = torch.as_tensor(tree["feat"], device=dev)
        left_t = torch.as_tensor(tree["left"], device=dev)
        right_t = torch.as_tensor(tree["right"], device=dev)
        node = torch.zeros(n, dtype=torch.long, device=dev)
        ar = torch.arange(n, device=dev)
        for _ in range(len(feat)):
            f = feat_t[node]
            leaf = f < 0
            if bool(leaf.all()):
                break
            bit = X[ar, f.clamp_min(0)] > 0
            node = torch.where(leaf, node, torch.where(bit, right_t[node], left_t[node]))
        return node
    node = np.zeros(n, np.int64)
    ar = np.arange(n)
    for _ in range(len(feat)):
        f = feat[node]
        leaf = f < 0
        if leaf.all():
            break
        bit = X[ar, np.maximum(f, 0)] > 0
        node = np.where(leaf, node, np.where(bit, tree["right"][node], tree["left"][node]))
    return node


def fit_boost(X: torch.Tensor, y: torch.Tensor, *, n_trees: int, max_leaves: int, min_leaf: int,
              lr: float, qscale: float | None, wbits: int,
              evalset=None, tag: str = "") -> tuple[list, list]:
    """SAMME. Returns (trees, alphas). If qscale is given, alpha is QUANTIZED to an integer in
    [1, 2^wbits-1] inside the loop and the *quantized* alpha drives the sample-weight update, so
    later trees correct the circuit-exact ensemble's residual rather than a float fiction."""
    dev = X.device
    N = X.shape[0]
    K = N_CLASSES
    w = torch.full((N,), 1.0 / N, device=dev)
    trees: list = []
    alphas: list = []
    score = None if evalset is None else torch.zeros(len(evalset[1]), K, device=dev)

    for t in range(n_trees):
        yoh = torch.zeros(N, K, device=dev)
        yoh[torch.arange(N, device=dev), y] = 1.0
        tree = build_tree(X, y, w, yoh * w[:, None], max_leaves, min_leaf)
        miss = (torch.as_tensor(tree["cls"], device=dev)[_route(tree, X)] != y).to(torch.float32)
        err = float((w * miss).sum() / w.sum())

        if err >= 1.0 - 1.0 / K:                          # worse than random: drop it
            w = torch.full((N,), 1.0 / N, device=dev)
            continue
        if err <= 1e-12:
            alpha = (math.log((1 - 1e-12) / 1e-12) + math.log(K - 1)) * lr
            reset = True
        else:
            alpha = (math.log((1 - err) / err) + math.log(K - 1)) * lr
            reset = False

        if qscale is not None:                            # the integer the CIRCUIT will use
            a_int = int(np.clip(round(alpha * qscale), 1, 2 ** wbits - 1))
            alpha_eff = a_int / qscale                    # ... and what training must react to
            keep = a_int
        else:
            alpha_eff = alpha
            keep = alpha

        if reset:
            w = torch.full((N,), 1.0 / N, device=dev)
        else:
            w = w * torch.exp(alpha_eff * miss)
            w = w / w.sum()

        trees.append(tree)
        alphas.append(keep)

        if evalset is not None:
            Xe, ye = evalset
            leaf_cls = torch.as_tensor(tree["cls"], device=dev)[_route(tree, Xe)]
            score[torch.arange(len(ye), device=dev), leaf_cls] += alpha_eff
            acc = 100.0 * float((score.argmax(1) == ye).float().mean())
            print(f"{tag}tree {t:4d} | err {err:.4f} a {alpha:5.2f} -> {keep} | val {acc:5.2f}",
                  flush=True)
    return trees, alphas


class Forest(Submission):
    def __init__(self, trees: int, leaves: int, wbits: int, bits: int,
                 min_leaf: int = 5, lr: float = 0.3):
        self.n_trees, self.max_leaves, self.wbits, self.bits = trees, leaves, wbits, bits
        self.min_leaf, self.lr = min_leaf, lr
        self.thresholds = even_thresholds(bits)
        self.trees: list = []
        self.w: list[int] = []

    # ---- the single source of truth for what a feature id MEANS -------------------------
    # feature id g <-> (pixel p, threshold index j) with g = p*k + j, matching the layout
    # hw.emit_thermometer documents. _encode() and _feat_expr() are the only two readers, and
    # neither open-codes the convention.
    def _split(self, g: int) -> tuple[int, int]:
        k = len(self.thresholds)
        return g // k, int(self.thresholds[g % k])

    def _encode(self, pix: np.ndarray) -> np.ndarray:
        """(N, 784) uint8 -> (N, 784*k) uint8 thermometer bits. Strict `>`, as in the Verilog."""
        cols = [(pix > int(t)).astype(np.uint8) for t in self.thresholds]
        return np.stack(cols, axis=2).reshape(len(pix), -1)

    def _feat_expr(self, g: int) -> str:
        p, t = self._split(g)
        if t == 127:                       # pix > 127 is bit 7 of the byte: a wire, 0 gates
            return f"pix[{p * PIXEL_BITS + 7}]"
        return f"(pix[{p * PIXEL_BITS} +: {PIXEL_BITS}] > 8'd{t})"

    # ---- exact rewrites, applied before emission ---------------------------------------
    def _collapse(self, tree: dict) -> dict:
        """If both children of a node are leaves of the SAME class, that node becomes that leaf.
        Exactly accuracy-neutral; the leaf-wise builder produces these constantly."""
        feat, left, right, cls = (tree[k].copy() for k in ("feat", "left", "right", "cls"))
        changed = True
        while changed:
            changed = False
            for n in range(len(feat)):
                if feat[n] < 0:
                    continue
                l, r = left[n], right[n]
                if feat[l] < 0 and feat[r] < 0 and cls[l] == cls[r]:
                    feat[n], left[n], right[n], cls[n] = -1, -1, -1, cls[l]
                    changed = True
        return {"feat": feat, "left": left, "right": right, "cls": cls}

    def _leaves_under(self, tree: dict) -> dict[int, set]:
        """Class set of the leaves under each node (post-order, iterative)."""
        feat, left, right, cls = tree["feat"], tree["left"], tree["right"], tree["cls"]
        out: dict[int, set] = {}
        stack = [(0, False)]
        while stack:
            n, done = stack.pop()
            if feat[n] < 0:
                out[n] = {int(cls[n])}
                continue
            if done:
                out[n] = out[int(left[n])] | out[int(right[n])]
            else:
                stack.append((n, True))
                stack.append((int(left[n]), False))
                stack.append((int(right[n]), False))
        return out

    # ---- training ----------------------------------------------------------------------
    def train(self, data: Mnist, *, device: str = "cpu", seed: int = 0) -> None:
        torch.manual_seed(seed)
        dev = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        X = torch.from_numpy(self._encode(data.train_x)).to(dev)
        y = torch.from_numpy(data.train_y).to(dev)
        Xv = torch.from_numpy(self._encode(data.val_x)).to(dev)
        yv = torch.from_numpy(data.val_y).to(dev)

        # pass 1: unquantized, only to see where alpha lands. Nothing here is hand-tuned.
        _, a0 = fit_boost(X, y, n_trees=self.n_trees, max_leaves=self.max_leaves,
                          min_leaf=self.min_leaf, lr=self.lr, qscale=None, wbits=self.wbits)
        p95 = float(np.percentile(np.asarray(a0, float), 95))
        qscale = (2 ** self.wbits - 1) / max(p95, 1e-9)
        print(f"[quant] alpha p95 {p95:.3f} -> wscale {qscale:.3f} "
              f"(alpha -> int in [1, {2 ** self.wbits - 1}])", flush=True)

        # pass 2: the real fit, with the circuit's integers in the loop
        trees, w = fit_boost(X, y, n_trees=self.n_trees, max_leaves=self.max_leaves,
                             min_leaf=self.min_leaf, lr=self.lr, qscale=qscale,
                             wbits=self.wbits, evalset=(Xv, yv))
        self.trees = [self._collapse(t) for t in trees]
        self.w = [int(a) for a in w]
        assert self.trees, "no tree survived boosting"
        assert min(self.w) >= 1, f"weights must be >= 1 (unsigned scores), got {min(self.w)}"

        nl = sum(int((t["feat"] < 0).sum()) for t in self.trees)
        acc = 100.0 * (self.predict(data.val_x) == data.val_y).mean()
        print(f"[forest] {len(self.trees)} trees, {nl:,} leaves, weights {self.w} "
              f"| val {acc:.2f}%", flush=True)

    # ---- the circuit's integers, in python ---------------------------------------------
    def _score_int(self, pix: np.ndarray) -> np.ndarray:
        """(N, 784) uint8 -> (N, 10) int64. This is EXACTLY what the emitted `score` holds: no
        float touches the decision path, so python and Verilog cannot drift apart."""
        F = self._encode(pix)
        s = np.zeros((len(pix), N_CLASSES), np.int64)
        ar = np.arange(len(pix))
        for tree, w in zip(self.trees, self.w):
            s[ar, tree["cls"][_route(tree, F)]] += w
        return s

    def predict(self, pix: np.ndarray) -> np.ndarray:
        return self._score_int(pix).argmax(1)          # ties -> lowest class, as the argmax emits

    def scores(self, pix: np.ndarray) -> np.ndarray:
        # one positive constant, the same for every class and image: a strictly increasing affine
        # map, so the argmax (ties included) is bit-for-bit the one predict() takes.
        return self._score_int(pix) / float(sum(self.w))

    # ---- emission ----------------------------------------------------------------------
    def emit_verilog(self) -> str:
        W = int(sum(self.w)).bit_length()
        assert sum(self.w) < 2 ** W, "score width too narrow -- would truncate silently"

        body: list[str] = []

        # features: ONLY the ones some node actually splits on. opt_clean would delete the rest,
        # but not emitting them keeps the AIG (and yosys's read time) small.
        used = sorted({int(f) for t in self.trees for f in t["feat"] if f >= 0})
        body.append(f"  // thermometer features actually used: {len(used)} of "
                    f"{N_PIXELS * len(self.thresholds)}")
        for g in used:
            body.append(f"  wire f{g} = {self._feat_expr(g)};")

        # per-tree reach network + class indicators
        vind: list[dict[int, str]] = []
        for ti, tree in enumerate(self.trees):
            feat, left, right = tree["feat"], tree["left"], tree["right"]
            nl = int((feat < 0).sum())
            body.append(f"  // tree {ti}: {nl} leaves, weight {self.w[ti]}")

            def reach(n: int) -> str:
                return "1'b1" if n == 0 else f"r{ti}_{n}"

            stack = [0]
            while stack:                                  # emit reach wires top-down
                n = stack.pop()
                if feat[n] < 0:
                    continue
                f = f"f{int(feat[n])}"
                for child, lit in ((int(left[n]), f"~{f}"), (int(right[n]), f)):
                    rhs = lit if n == 0 else f"{reach(n)} & {lit}"
                    body.append(f"  wire {reach(child)} = {rhs};")
                    stack.append(child)

            lu = self._leaves_under(tree)
            cls = tree["cls"]

            def class_ind(n: int, c: int) -> str | None:
                if feat[n] < 0:
                    return reach(n) if int(cls[n]) == c else None
                if lu[n] == {c}:                          # whole subtree is class c: reuse reach
                    return reach(n)
                if c not in lu[n]:
                    return None
                a = class_ind(int(left[n]), c)
                b = class_ind(int(right[n]), c)
                if a is None:
                    return b
                if b is None:
                    return a
                return f"({a} | {b})"

            vi: dict[int, str] = {}
            for c in range(N_CLASSES):
                e = class_ind(0, c)
                if e is not None:
                    body.append(f"  wire v{ti}_{c} = {e};")
                    vi[c] = f"v{ti}_{c}"
            vind.append(vi)

        # head: bit-plane popcount. A zero weight-bit costs nothing at all.
        body.append(f"  // head: per-class bit-plane popcount, {W}-bit unsigned scores")
        body.append(f"  logic [{W - 1}:0] score [0:{N_CLASSES - 1}];")
        for c in range(N_CLASSES):
            planes = []
            for b in range(self.wbits):
                terms = [vind[t][c] for t in range(len(self.trees))
                         if (self.w[t] >> b) & 1 and c in vind[t]]
                if not terms:
                    continue
                # every plane is W bits wide; ABC deletes the dead MSBs at zero cost, and the
                # alternative (a tight per-plane width) is a truncation bug waiting to happen
                body.append(f"  logic [{W - 1}:0] p{b}_c{c};")
                body.append(f"  assign p{b}_c{c} = {' + '.join(terms)};")
                planes.append(f"{1 << b} * p{b}_c{c}" if b else f"p{b}_c{c}")
            rhs = " + ".join(planes) if planes else f"{W}'d0"
            body.append(f"  assign score[{c}] = {rhs};")

        # argmax: strict >, ascending c -> ties to the lowest class (hw.emit_popcount_argmax)
        body.append(f"  logic [{W - 1}:0] best;")
        body.append("  always_comb begin")
        body.append("    best = score[0];")
        body.append("    cls  = 4'd0;")
        for c in range(1, N_CLASSES):
            body.append(f"    if (score[{c}] > best) begin best = score[{c}]; "
                        f"cls = 4'd{c}; end")
        body.append("  end")

        nl = sum(int((t["feat"] < 0).sum()) for t in self.trees)
        return (f"// generated by records/sbuehrer/forest -- {len(self.trees)} SAMME trees, "
                f"{nl} leaves,\n"
                f"// {len(self.thresholds)} thermometer bits/pixel, {self.wbits}-bit tree "
                f"weights, {W}-bit scores\n"
                f"module top (input [{N_PIXELS * PIXEL_BITS - 1}:0] pix, "
                f"output logic [3:0] cls);\n\n"
                + "\n".join(body) + "\nendmodule\n")


def build(**point) -> Submission:
    return Forest(**point)
