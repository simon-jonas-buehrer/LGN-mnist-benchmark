"""tao: a deep network of decision-tree nodes, trained by gradient-targeted refitting.

PHASE 1 PROTOTYPE. No Verilog yet -- this file answers "does it learn?", and `estimate_gates()`
places the answer on the benchmark's x-axis well enough to know whether it is worth emitting.

The model is a layered net in which every node is a small decision tree with a FULL receptive
field over the previous layer:

    thermometer encoder -> M0 tree-nodes -> M1 tree-nodes -> ... -> popcount -> argmax

so the wiring is chosen by information gain over ALL candidate bits, rather than (as in the
backprop record) by gradient over 8 randomly drawn candidates per gate input. The tree builder IS
the wiring optimizer. That is the claim this record exists to test.

Two facts make the combination work, and both are exact rather than approximate:

  The gradient w.r.t. an input bit is a finite difference, and is automatically sparse.
  A tree's output is multilinear in the bits it reads:

      out(x) = sum_leaves v_l * prod_{(f,s) on path to l} lit(x_f, s)

  On binary inputs the partial derivative of a multilinear function IS a finite difference, with
  no truncation error at all:

      d out / d x_f = out(x_f = 1) - out(x_f = 0)
      flipping bit f moves the output by exactly (1 - 2*x_f) * d out / d x_f

  Every off-path product contains a zero factor, so only the <=D features on the path ACTUALLY
  TAKEN receive gradient. "Only send gradients to the inputs this node's tree used" is not
  engineered here; it falls out of the algebra. LutLayer in the backprop record already uses the
  2-input case of this same form.

  That identity has one precondition, and `proto.py --gradcheck` is what found it: NO ROOT-TO-LEAF
  PATH MAY TEST THE SAME FEATURE TWICE. A repeat puts x_f * x_f in the product, and x^2 = x is
  true as a function on {0,1} but false as a derivative -- a contradictory repeat gives a term
  that is identically zero yet has gradient 1 - 2x_f, so the "gradient" would point somewhere the
  output cannot go. So the invariant is enforced structurally, in the init and in the refit
  (`_distinct_randint`, and the ancestor mask in `refit_layer`). It costs nothing: a second test
  of a feature already decided on the path is a redundant gate anyway.

  Gradient and tree-refit are one update, split by what each can move.
  Gradient moves the leaf values (continuous latents, straight-through). It cannot move a split
  feature -- that is a discrete jump. So the loop alternates: gradient tunes the leaves, and a
  periodic greedy refit moves the wires, fitting each node to a target read off its OWN gradient,

      target[b, m] = 1[g[b, m] < 0]        which way should this bit have gone
      weight[b, m] = |g[b, m]|             how much did it matter

  which is a weighted binary classification problem per node -- exactly what the forest record's
  builder solves, and over binary features a split search is a single GEMM. Here that GEMM is
  batched over every node and every cell of a layer at once (`refit_layer`), instead of looped.

Named after Tree Alternating Optimization, the closest existing method: alternate between fitting
a node's tree and the signal it is fit against.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from mnistbench.data import Mnist, N_CLASSES, N_PIXELS
from mnistbench.hw import even_thresholds

# ==========================================================================================
# Straight-through binarisation. Same sin-STE as records/sbuehrer/backprop: sin is periodic, so
# a latent never saturates and there is always a gradient toward the nearest 0/1 basin.
# ==========================================================================================
LATENT = 1.0  # |leaf latent| set by a refit. sin(1)=0.84 (firmly binary), cos(1)=0.54 (live grad)


def hard_bit(z: torch.Tensor) -> torch.Tensor:
    return (torch.sin(z) > 0).to(z.dtype)


def ste_bit(z: torch.Tensor) -> torch.Tensor:
    soft = 0.5 + 0.5 * torch.sin(z)
    return hard_bit(z) + (soft - soft.detach())


# ==========================================================================================
# One layer: `width` complete depth-D decision trees, every one reading all `n_in` input bits.
#
# Layout, fixed once and relied on by the forward pass, the refit, the reference router and
# (later) the emitter:
#
#   feat  (width, 2^D - 1)  split feature per internal slot, HEAP ORDER: level d owns slots
#                           2^d - 1 .. 2^(d+1) - 2, and cell c of level d is slot 2^d - 1 + c.
#   leaf  (width, 2^D)      latent leaf value; leaf l is reached by the path whose level-d bit
#                           is (l >> (D-1-d)) & 1, i.e. cell = 2*cell + bit descending.
# ==========================================================================================
def _distinct_randint(width: int, k: int, n: int, g: torch.Generator) -> torch.Tensor:
    """(width, k) random ids in [0, n), distinct WITHIN each row.

    Distinct across all slots is stronger than the no-repeat-per-path invariant the gradient
    needs, but it is simpler and costs nothing. k << n, so rejection converges in a few rounds.
    """
    if n <= k:
        raise ValueError(f"need more than {k} input bits to give a depth-{k} tree distinct splits")
    out = torch.randint(n, (width, k), generator=g)
    for _ in range(64):
        s, _ = out.sort(1)
        dup = torch.zeros_like(out, dtype=torch.bool)
        dup[:, 1:] = s[:, 1:] == s[:, :-1]
        rows = dup.any(1)
        if not bool(rows.any()):
            break
        out[rows] = torch.randint(n, (int(rows.sum()), k), generator=g)
    return out


class TreeLayer(torch.nn.Module):
    def __init__(self, n_in: int, width: int, depth: int, g: torch.Generator) -> None:
        super().__init__()
        self.n_in, self.width, self.depth = n_in, width, depth
        self.n_slots = (1 << depth) - 1
        self.n_leaf = 1 << depth
        # random structure; the dichotomy init overwrites it before any gradient step. Distinct
        # per node even so: the no-repeat-per-path invariant must hold at every instant, not only
        # after the first refit.
        self.register_buffer("feat", _distinct_randint(width, self.n_slots, n_in, g))
        self.leaf = torch.nn.Parameter(torch.randn(width, self.n_leaf, generator=g))

    def path_repeats(self) -> int:
        """How many (node, leaf) paths test some feature twice. Must always be 0 -- see module
        docstring; this is what makes the input gradient an exact finite difference."""
        bad = 0
        for l in range(self.n_leaf):
            seen, cell = [], 0
            for d in range(self.depth):
                seen.append(self.feat[:, ((1 << d) - 1) + cell])
                cell = cell * 2 + ((l >> (self.depth - 1 - d)) & 1)
            p = torch.stack(seen, 1)                                   # (width, depth)
            bad += int(((p[:, :, None] == p[:, None, :]).sum(-1) > 1).any(1).sum())
        return bad

    # ---- forward: multilinear, exact-hard on binary input ------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_in) exact 0.0/1.0 -> (B, width) exact 0.0/1.0.

        `reach` is the multilinear extension of the leaf indicators. On hard input it is exactly
        one-hot over leaves, so the output is exactly the routed leaf bit -- the forward pass is
        already the circuit. On the backward pass the products are what turn into the finite
        differences described in the module docstring.
        """
        B = x.shape[0]
        reach = x.new_ones(B, self.width, 1)
        lo = 0
        for d in range(self.depth):
            k = 1 << d
            f = x[:, self.feat[:, lo : lo + k]]  # (B, width, k)
            reach = torch.stack([reach * (1 - f), reach * f], -1).reshape(B, self.width, 2 * k)
            lo += k
        return (reach * ste_bit(self.leaf)).sum(-1)

    # ---- hard routing, used by the refit (no autograd, no reach tensor) -----------------
    # These take `feat`/`leafbit` explicitly and size themselves from it, so the refit can run
    # them on a SUBSET of the layer's nodes without building a second module.
    def cells(self, x: torch.Tensor, feat: torch.Tensor, upto: int) -> torch.Tensor:
        """(B, n_in) -> (B, len(feat)) cell index in [0, 2^upto), descending with the GIVEN feat."""
        B, M = x.shape[0], feat.shape[0]
        base = torch.arange(M, device=x.device) * self.n_slots
        cell = torch.zeros(B, M, dtype=torch.long, device=x.device)
        flat = feat.reshape(-1)
        for d in range(upto):
            slot = ((1 << d) - 1) + cell                       # (B, M)
            fi = flat[base + slot]                             # (B, M) feature id
            cell = cell * 2 + torch.gather(x, 1, fi).long()
        return cell

    def hard_out(self, x: torch.Tensor, feat: torch.Tensor, leafbit: torch.Tensor) -> torch.Tensor:
        """The layer's exact output for an arbitrary (feat, leafbit) pair -- what accept/reject
        compares, and what the reference router must reproduce."""
        cell = self.cells(x, feat, self.depth)
        base = torch.arange(feat.shape[0], device=x.device) * self.n_leaf
        return leafbit.reshape(-1)[base + cell]

    def leafbit(self) -> torch.Tensor:
        return hard_bit(self.leaf)


# ==========================================================================================
# The refit: rebuild every tree in a layer, level by level, all nodes and all cells in ONE GEMM
# per level. This is forest.best_split's `wyoh[idx].t() @ Xg` batched instead of looped.
# ==========================================================================================
def _gini(cnt0: torch.Tensor, cnt1: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Weighted-Gini split score, summed over the two sides. cnt* are (..., 2, F) class weights.
    Same objective as forest.best_split: sum_y n_y^2 / sum_y n_y, maximised."""
    return (cnt0.pow(2).sum(-2) / cnt0.sum(-2).clamp_min(eps)
            + cnt1.pow(2).sum(-2) / cnt1.sum(-2).clamp_min(eps))


def worst_nodes(layer: TreeLayer, X: torch.Tensor, tgt: torch.Tensor, w: torch.Tensor,
                frac: float) -> torch.Tensor:
    """The `frac` of nodes whose current tree fits its target worst -- the ones with the most to
    gain from being rebuilt. Restricting a refit round to these is the trust region that makes the
    alternation work at all: the target is a LINEARISATION of the loss around the current bits, so
    rebuilding every node at once lands far outside where that linearisation means anything."""
    out = layer.hard_out(X, layer.feat, layer.leafbit())
    err = (w * (out - tgt.to(X.dtype)).abs()).sum(0)               # (width,)
    k = max(1, int(round(frac * layer.width)))
    return err.topk(k).indices


def refit_layer(layer: TreeLayer, X: torch.Tensor, tgt: torch.Tensor, w: torch.Tensor, *,
                sub: torch.Tensor | None = None, mtry: int = 0, chunk: int = 256,
                accept: bool = True, gen: torch.Generator | None = None) -> float:
    """Refit the trees in `layer` (or just those in `sub`) to their own weighted binary target.
    Returns the fraction of candidate nodes whose new tree was kept.

    X    (B, n_in)  the layer's input bits, exact 0.0/1.0
    tgt  (B, width) 0/1 target for each node's output
    w    (B, width) non-negative sample weight for each node
    sub  (k,)       node ids to rebuild; None = all (the dichotomy init)

    `mtry` (0 = all) restricts the candidate features of a node-chunk to a random subset. It is a
    tractability knob and a DIVERSITY knob: greedy Gini is deterministic, so without per-node
    variation every node fed the same target rebuilds the same tree. `fit` also bags the weights
    per node, which is the other half of that fix.

    A node keeps its new tree only if that tree lowers weighted target error on this very sample.
    That is necessary but NOT sufficient -- the target is a linearisation, so `fit` additionally
    reverts the whole layer if the true loss got worse. Only nodes that are actually rebuilt have
    their leaves reset; everything else keeps the values the gradient has tuned.
    """
    dev, B, D = X.device, X.shape[0], layer.depth
    if mtry and mtry <= D:  # else a cell could run out of non-ancestor candidates
        raise ValueError(f"mtry={mtry} must exceed depth {D}")
    if sub is None:
        sub = torch.arange(layer.width, device=dev)
    M = sub.numel()
    tgt = tgt[:, sub].to(X.dtype)
    w = w[:, sub]
    new_feat = torch.zeros(M, layer.n_slots, dtype=layer.feat.dtype, device=dev)

    for d in range(D):
        K = 1 << d                                   # cells at this level
        cell = layer.cells(X, new_feat, d)           # (B, M), from the levels already rebuilt
        col_base = (cell * 2 + tgt.long())           # (B, M) offset within a node's block
        c_ar = torch.arange(K, device=dev)
        for m0 in range(0, M, chunk):
            m1 = min(m0 + chunk, M)
            Mc = m1 - m0
            if mtry and mtry < layer.n_in:           # candidate subset, shared by this chunk
                cand = torch.randperm(layer.n_in, generator=gen, device=dev)[:mtry]
                Xc = X[:, cand]
            else:
                cand = torch.arange(layer.n_in, device=dev)
                Xc = X

            # G[b, (m,c,y)] = w * [cell==c] * [tgt==y]. One column per (node, cell, class), and
            # for a fixed b each node writes exactly one column, so the scatter never collides.
            col = torch.arange(Mc, device=dev) * (K * 2) + col_base[:, m0:m1]
            G = torch.zeros(B, Mc * K * 2, device=dev, dtype=X.dtype)
            G.scatter_(1, col, w[:, m0:m1].contiguous())

            cnt1 = (G.t() @ Xc).reshape(Mc, K, 2, -1)          # weight with bit==1, per class
            tot = G.sum(0).reshape(Mc, K, 2, 1)
            cnt0 = tot - cnt1
            score = _gini(cnt0, cnt1)                          # (Mc, K, F_cand)
            # a split that sends (almost) nothing one way is not a split
            dead = (cnt0.sum(-2) < 1e-9) | (cnt1.sum(-2) < 1e-9)

            # never re-test a feature this cell's path already decided. A weighted split search
            # rules those out on its own -- an ancestor feature is constant on the cell, so one
            # side is empty and `dead` already catches it -- but ONLY while the cell still has
            # weight on it. An unreachable or zero-weight cell has every candidate dead, and the
            # fallback must still respect the invariant, so mask ancestors explicitly.
            if d:
                anc = torch.stack([new_feat[m0:m1][:, ((1 << dd) - 1) + (c_ar >> (d - dd))]
                                   for dd in range(d)], -1)    # (Mc, K, d)
                banned = (cand.view(1, 1, -1) == anc.unsqueeze(-1)).any(-2)
                dead = dead | banned
            score = score.masked_fill(dead, float("-inf"))

            best = score.argmax(-1)                            # (Mc, K)
            alive = score.gather(-1, best[..., None]).squeeze(-1) > float("-inf")
            fallback = ((~banned).to(torch.uint8).argmax(-1) if d
                        else torch.zeros_like(best))           # first non-ancestor candidate
            slot = c_ar + (K - 1)
            new_feat[m0:m1, slot] = cand[torch.where(alive, best, fallback)]

    # leaves: weighted majority of the target in each leaf cell
    leaf_cell = layer.cells(X, new_feat, D)                    # (B, M)
    pos = torch.zeros(M, layer.n_leaf, device=dev, dtype=X.dtype)
    neg = torch.zeros_like(pos)
    pos.scatter_add_(1, leaf_cell.t(), (w * tgt).t().contiguous())
    neg.scatter_add_(1, leaf_cell.t(), (w * (1 - tgt)).t().contiguous())
    new_leafbit = (pos > neg).to(X.dtype)

    if accept:
        old = layer.hard_out(X, layer.feat[sub], hard_bit(layer.leaf[sub]))
        new = layer.hard_out(X, new_feat, new_leafbit)
        keep = (w * (new - tgt).abs()).sum(0) < (w * (old - tgt).abs()).sum(0)
    else:
        keep = torch.ones(M, dtype=torch.bool, device=dev)

    idx = sub[keep]
    layer.feat[idx] = new_feat[keep]
    # only a node that was actually rebuilt loses its leaves; the rest keep what the gradient
    # tuned. LATENT is binary but unsaturated, so the gradient can still move them afterwards.
    with torch.no_grad():
        layer.leaf[idx] = torch.where(new_leafbit[keep] > 0, LATENT, -LATENT)
    return float(keep.float().mean())


# ==========================================================================================
# The network
# ==========================================================================================
class TaoNet(torch.nn.Module):
    """Encoder + tree layers + the benchmark's popcount/argmax readout.

    Readout is hw.emit_popcount_argmax's function -- contiguous equal groups, popcount, argmax
    with ties to the lowest class -- identical to backprop/dfa/hebbian, so the comparison across
    records is at matched readout. Receptive field is the previous layer only.
    """

    def __init__(self, bits: int, widths: tuple[int, ...], depth: int, seed: int = 0) -> None:
        super().__init__()
        if widths[-1] % N_CLASSES:
            raise ValueError(f"readout width {widths[-1]} must be divisible by {N_CLASSES}")
        self.bits, self.widths, self.depth = bits, tuple(widths), depth
        self.thresholds = even_thresholds(bits)
        g = torch.Generator().manual_seed(seed)
        n_in = N_PIXELS * bits
        self.layers = torch.nn.ModuleList()
        for w in widths:
            self.layers.append(TreeLayer(n_in, w, depth, g))
            n_in = w

    def encode(self, pix: torch.Tensor) -> torch.Tensor:
        """(N, 784) uint8 -> (N, 784*bits) float bits, laid out exactly as hw.emit_thermometer."""
        t = torch.tensor(self.thresholds, device=pix.device, dtype=torch.int16)
        return (pix.to(torch.int16).unsqueeze(-1) > t).reshape(pix.shape[0], -1).float()

    def activations(self, pix: torch.Tensor, retain: bool = False) -> list[torch.Tensor]:
        """[encoder bits, layer0 out, layer1 out, ...]. With retain=True every layer output keeps
        its .grad after backward, which is the pseudo-target the refit is fit against."""
        acts = [self.encode(pix)]
        for lay in self.layers:
            h = lay(acts[-1])
            if retain:
                h.retain_grad()
            acts.append(h)
        return acts

    def head(self, last: torch.Tensor) -> torch.Tensor:
        g = self.widths[-1] // N_CLASSES
        return last.reshape(last.shape[0], N_CLASSES, g).sum(-1) / math.sqrt(g)

    def forward(self, pix: torch.Tensor) -> torch.Tensor:
        return self.head(self.activations(pix)[-1])

    @torch.no_grad()
    def votes(self, pix: torch.Tensor) -> torch.Tensor:
        """(N, 10) per-class firing fraction in [0, 1] -- the readout just before the argmax."""
        last = self.activations(pix)[-1]
        g = self.widths[-1] // N_CLASSES
        return last.reshape(last.shape[0], N_CLASSES, g).mean(-1)


# ==========================================================================================
# The reference router: pure numpy, no autograd, no multilinear form. The torch forward must
# reproduce it BIT FOR BIT -- that is the local stand-in for the harness's predict()-vs-netlist
# check, and it catches child-ordering and heap-indexing mistakes immediately.
# ==========================================================================================
def route_numpy(x: np.ndarray, feat: np.ndarray, leafbit: np.ndarray) -> np.ndarray:
    """(B, F) uint8 bits -> (B, M) uint8 bits, by walking each tree one level at a time."""
    B, M = x.shape[0], feat.shape[0]
    S, L = feat.shape[1], leafbit.shape[1]
    depth = int(math.log2(L))
    base = np.arange(M) * S
    cell = np.zeros((B, M), np.int64)
    flat = feat.reshape(-1)
    for d in range(depth):
        fi = flat[base + ((1 << d) - 1) + cell]                # (B, M)
        cell = cell * 2 + x[np.arange(B)[:, None], fi]
    return leafbit.reshape(-1)[np.arange(M) * L + cell].astype(np.uint8)


def predict_numpy(net: TaoNet, pix: np.ndarray) -> np.ndarray:
    """The whole model in numpy: encode, route every layer, popcount, argmax (ties -> lowest)."""
    thr = np.asarray(net.thresholds, np.int16)
    h = (pix.astype(np.int16)[:, :, None] > thr).reshape(len(pix), -1).astype(np.uint8)
    for lay in net.layers:
        h = route_numpy(h, lay.feat.cpu().numpy(), lay.leafbit().cpu().numpy().astype(np.uint8))
    g = net.widths[-1] // N_CLASSES
    return h.reshape(len(pix), N_CLASSES, g).sum(-1).argmax(1)


# ==========================================================================================
# Area estimate. Rough and PRE-ABC: it prices one node in isolation and cannot see the sharing
# ABC finds between nodes. Its job is the order of magnitude -- is this point near forest-m or
# near bitnet? -- not the leaderboard number, which only yosys may produce.
# ==========================================================================================
# Constants calibrated against the two MEASURED backprop points that share our encoder and our
# readout: xs (bits=1, 480 gates, 160 readout bits -> 1,913 GE) and s (bits=1, 1920 gates, 640
# readout bits -> 7,514 GE). Both have a free encoder (bits=1 is the single threshold 127, which
# is bit 7 of the byte -- a wire), so those two points isolate the head:
#     480a + 160b + c = 1913,  1920a + 640b + c = 7514
# and at a ~= 2 GE for an average 2-input gate they give b ~= 5.7, c ~= 50.
AND_GE = 1.5       # a 2-input AND/OR cell in sky130
MUX_GE = 3.0       # 2:1 mux with two live data inputs
READOUT_GE = 5.7   # per readout bit: its share of the popcount tree and the argmax
FIXED_GE = 50.0

_CONST, _WIRE, _GATE = 0, 1, 2


def thresh_ge(t: int) -> float:
    """Area of `pix > t` for a uint8 pixel, in GE.

    Not a generic 8-bit comparator: `pix > t` only has to look at the bits above the trailing run
    of ones in t+1. hw.even_thresholds deliberately lands on those boundaries -- `pix > 127` is
    bit 7 of the byte, a WIRE costing nothing, and `pix > 63` is `pix[7] | pix[6]`, one gate. At
    bits=3 the whole encoder is about one gate per used bit, not the handful a naive comparator
    would suggest, and getting this wrong overstates a small circuit's area several times over.
    """
    c = int(t) + 1
    lsb = (c & -c).bit_length() - 1          # lowest set bit of t+1
    return max(0, 7 - lsb) * AND_GE


def _prune_node(feat: np.ndarray, leafbit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bottom-up cost of every node in a layer, and which slots survive.

    A subtree whose leaves are all equal collapses to a constant and costs nothing; a node whose
    two children are opposite constants collapses to the literal itself (a wire or an inverter);
    a node with one constant child is an AND/OR, one gate. Anything else is a mux. This is the
    saving a leaf-wise builder would have got from an incomplete tree, recovered at emission.

    Returns (cost per node, live-slot mask), the cost in GE.
    """
    M, L = leafbit.shape
    depth = int(math.log2(L))
    kind = np.full((M, L), _CONST, np.int8)
    val = leafbit.astype(np.int8)
    cost = np.zeros((M, L), np.float64)
    live = np.zeros_like(feat, bool)

    for d in range(depth - 1, -1, -1):
        K = 1 << d
        kl, kr = kind[:, 0::2], kind[:, 1::2]
        vl, vr = val[:, 0::2], val[:, 1::2]
        cl, cr = cost[:, 0::2], cost[:, 1::2]
        both_const = (kl == _CONST) & (kr == _CONST)
        same = both_const & (vl == vr)
        flip = both_const & (vl != vr)                          # -> the literal: a wire
        one_const = (kl == _CONST) ^ (kr == _CONST)
        nk = np.where(same, _CONST, np.where(flip, _WIRE, _GATE)).astype(np.int8)
        nc = np.where(same, 0.0,
                      np.where(flip, 0.0,
                               np.where(one_const, cl + cr + AND_GE, cl + cr + MUX_GE)))
        kind, val, cost = nk, np.where(same, vl, 0).astype(np.int8), nc
        live[:, K - 1 : 2 * K - 1] = ~same                      # a collapsed slot reads nothing
    return cost[:, 0], live


def estimate_gates(net: TaoNet) -> dict:
    """Pre-ABC gate estimate for the whole circuit, plus the counts it is built from."""
    total, live_slots = 0.0, 0
    per_layer = []
    used_enc: set[int] = set()
    for li, lay in enumerate(net.layers):
        feat = lay.feat.cpu().numpy()
        cost, live = _prune_node(feat, lay.leafbit().cpu().numpy())
        if li == 0:
            used_enc = set(feat[live].tolist())
        per_layer.append({"width": lay.width, "ge": round(float(cost.sum())),
                          "live_slots": int(live.sum()), "free_nodes": int((cost == 0).sum())})
        total += float(cost.sum())
        live_slots += int(live.sum())

    enc = sum(thresh_ge(net.thresholds[f % net.bits]) for f in used_enc)
    head = READOUT_GE * net.widths[-1] + FIXED_GE
    return {"ge_est": round(total + enc + head), "logic": round(total), "encoder": round(enc),
            "head": round(head), "live_slots": live_slots, "enc_bits_used": len(used_enc),
            "layers": per_layer}


# ==========================================================================================
# The optimizer: dichotomy init, then alternate gradient (leaves) and refit (wires).
# ==========================================================================================
def _dichotomy_targets(y: torch.Tensor, width: int, g: torch.Generator) -> torch.Tensor:
    """(B, width) 0/1. Node m is fit to a random class dichotomy c_m in {0,1}^10 -- an error-
    correcting output code. Greedy Gini is deterministic, so without this every node in a layer
    would build the SAME tree. Codes are redrawn per layer."""
    code = torch.randint(2, (width, N_CLASSES), generator=g, device=y.device)
    while True:  # a degenerate all-0/all-1 code carries no signal; redraw those rows
        bad = (code.sum(1) == 0) | (code.sum(1) == N_CLASSES)
        if not bool(bad.any()):
            break
        code[bad] = torch.randint(2, (int(bad.sum()), N_CLASSES), generator=g, device=y.device)
    return code[:, y].t().float()


@torch.no_grad()
def dichotomy_init(net: TaoNet, x: torch.Tensor, y: torch.Tensor, *, mtry: int, chunk: int,
                   gen: torch.Generator) -> None:
    """Give every layer an informative starting tree: fit layer 0 to random class dichotomies of
    the label, forward it, fit layer 1 to fresh dichotomies over layer 0's bits, and so on."""
    h = net.encode(x)
    for li, lay in enumerate(net.layers):
        tgt = _dichotomy_targets(y, lay.width, gen)
        w = torch.ones_like(tgt)
        frac = refit_layer(lay, h, tgt, w, mtry=mtry, chunk=chunk, accept=False, gen=gen)
        agree = float((lay.hard_out(h, lay.feat, lay.leafbit()) == tgt).float().mean())
        print(f"  [init] layer {li}: {lay.width} nodes, dichotomy agreement {agree * 100:.1f}%",
              flush=True)
        h = lay.hard_out(h, lay.feat, lay.leafbit())
        del frac


def _grad_targets(net: TaoNet, x: torch.Tensor, y: torch.Tensor, bag: float,
                  gen: torch.Generator) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """One backward pass -> (input bits, target, weight) per layer.

    target = 1[g < 0]: g is the exact change in the loss from flipping this bit (the network is
    multilinear in the hidden bits, so the only approximation is the softmax at the very end), so
    the sign says which way the bit should have gone and |g| says how much it mattered.

    `bag` bootstraps the weights per node. With mtry it is what keeps nodes fed identical targets
    -- notably every node inside one readout group -- from rebuilding identical trees.
    """
    acts = net.activations(x, retain=True)
    loss = torch.nn.functional.cross_entropy(net.head(acts[-1]), y)
    net.zero_grad(set_to_none=True)
    loss.backward()
    out = []
    for li, lay in enumerate(net.layers):
        g = acts[li + 1].grad
        w = g.abs()
        w = w / w.mean().clamp_min(1e-30)                      # keep the Gini scale sane
        if bag > 0:
            w = w * (torch.rand(w.shape, generator=gen, device=w.device) < bag).float()
        out.append((acts[li].detach(), (g < 0).float(), w.detach()))
    return out


@torch.no_grad()
def _loss(net: TaoNet, x: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.nn.functional.cross_entropy(net(x), y))


def fit(net: TaoNet, data: Mnist, *, device: str = "cpu", seed: int = 0, epochs: int = 60,
        batch: int = 128, lr: float = 0.05, patience: int = 15, refit_every: int = 2,
        refit_rows: int = 2048, refit_frac: float = 0.1, mtry: int = 1024, chunk: int = 256,
        bag: float = 0.7, do_grad: bool = True, do_refit: bool = True,
        log_every: int = 1) -> float:
    """Train. Returns the best val accuracy, and leaves `net` holding that state.

    do_grad / do_refit exist for the ablation: the alternation has to beat both halves of itself
    or it is not earning its complexity.
    """
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed + 1)
    x = torch.from_numpy(np.ascontiguousarray(data.train_x)).to(device)
    y = torch.from_numpy(data.train_y).to(device)
    vx = torch.from_numpy(np.ascontiguousarray(data.val_x)).to(device)
    vy = torch.from_numpy(data.val_y).to(device)

    print(f"[init] dichotomy init on {refit_rows} rows", flush=True)
    idx = torch.randperm(x.shape[0], generator=gen, device=device)[:refit_rows]
    dichotomy_init(net, x[idx], y[idx], mtry=mtry, chunk=chunk, gen=gen)

    def val_acc() -> float:
        with torch.no_grad():
            ch = 1024
            ok = sum(int((net(vx[i : i + ch]).argmax(1) == vy[i : i + ch]).sum())
                     for i in range(0, vx.shape[0], ch))
        return 100.0 * ok / vx.shape[0]

    best, best_state, best_ep = val_acc(), {k: v.detach().clone()
                                            for k, v in net.state_dict().items()}, -1
    print(f"[init] val {best:.2f}%", flush=True)

    opt = torch.optim.Adam([lay.leaf for lay in net.layers], lr=lr) if do_grad else None
    t0 = time.time()
    for ep in range(epochs):
        if do_grad:
            perm = torch.randperm(x.shape[0], generator=gen, device=device)
            for i in range(0, x.shape[0], batch):
                b = perm[i : i + batch]
                loss = torch.nn.functional.cross_entropy(net(x[b]), y[b])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        if do_refit and (ep + 1) % refit_every == 0:
            idx = torch.randperm(x.shape[0], generator=gen, device=device)[:refit_rows]
            rx, ry = x[idx], y[idx]
            base = _loss(net, rx, ry)
            note, moved = [], False
            # One layer at a time, each gated on the TRUE loss. The per-node target is only a
            # linearisation of that loss, so "better on the target" has to be checked against the
            # thing we actually minimise -- otherwise the refit reliably undoes the gradient.
            for li, (X, tgt, w) in enumerate(_grad_targets(net, rx, ry, bag, gen)):
                lay = net.layers[li]
                saved = (lay.feat.clone(), lay.leaf.detach().clone())
                sub = worst_nodes(lay, X, tgt, w, refit_frac)
                frac = refit_layer(lay, X, tgt, w, sub=sub, mtry=mtry, chunk=chunk, gen=gen)
                after = _loss(net, rx, ry)
                if after < base:
                    note.append(f"L{li} {frac * 100:.0f}%/{sub.numel()} {base:.3f}->{after:.3f}")
                    base, moved = after, True
                else:                                   # outside the trust region: put it back
                    lay.feat.copy_(saved[0])
                    with torch.no_grad():
                        lay.leaf.copy_(saved[1])
                    note.append(f"L{li} reverted")
            if do_grad and moved:  # the structure moved, so Adam's moments are stale
                opt = torch.optim.Adam([lay.leaf for lay in net.layers], lr=lr)
            print(f"  [refit] epoch {ep + 1}: " + "  ".join(note), flush=True)

        acc = val_acc()
        if acc > best:
            best, best_ep = acc, ep
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if ep % log_every == 0 or ep == epochs - 1:
            print(f"  epoch {ep + 1:3d}/{epochs}  val {acc:.2f}%  "
                  f"(best {best:.2f}% @ {best_ep + 1})  {time.time() - t0:.0f}s", flush=True)
        if ep - best_ep >= patience:
            print(f"  early stop at epoch {ep + 1}: no gain since {best_ep + 1}", flush=True)
            break

    net.load_state_dict(best_state)
    return best
