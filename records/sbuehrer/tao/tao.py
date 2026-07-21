"""tao: a binary network of decision trees, trained by a local binary error signal.

PHASE 1 PROTOTYPE. No Verilog yet -- this answers "does it learn?", and `estimate_gates()` places
the answer on the benchmark's x-axis well enough to know whether it is worth emitting.

Every node is a small decision tree with a FULL receptive field over the previous layer, so the
wiring is chosen by looking at every candidate bit rather than, as in the backprop record, by
gradient over 8 randomly drawn candidates per gate input. The tree builder IS the wiring optimizer.

    thermometer encoder -> M0 tree-nodes -> M1 tree-nodes -> ... -> popcount -> argmax

NOTHING HERE IS CONTINUOUS. There is no gradient, no float, no loss surface. One pass forward
routes trees; one pass back carries a single BIT per node. The whole rule is:

  forward   route each sample through each tree. Bitwise: a leaf indicator is a conjunction of
            literals, so a batch is AND/OR over 64-sample words (`mnistbench/netlist.py` already
            simulates netlists this way), and this is the same work the emitted circuit does.

  error     at the readout, a node in class c's group should fire exactly when c is the answer.
            Its error bit is `out XOR should`. Nothing else is needed to start.

  backward  a node with an error bit asks which of the bits it reads would fix it. On a binary
            tree over binary inputs that is EXACT -- flip the bit, fall into the sibling subtree,
            route the rest of the way with the real bits, read the leaf. No derivative exists or
            is needed. A consumer votes +1 on a bit whose flip would fix it and -1 on one whose
            flip would break it while it is currently right; the producing node sums the votes it
            receives and takes the sign. Signals on the wire are bits; the accumulator is local to
            the node, which is the only place it could live in hardware.

  update    a node compares its own error signal against ALL candidate input bits over the batch
            and changes ONE decision -- which input it tests, and hence which rule -- or the best
            `topk` of its 2^D - 1 decisions. Leaves always follow the wiring, by majority.

So a hardware implementation is a worker that visits a node, reads its error bits and its
candidate inputs, and rewrites one decision. On a GPU the same thing runs bitpacked, a whole batch
per word, which is why the candidate search below is written as a GEMM: X is binary and the
scatter matrix is one-hot, so the "matmul" is a bank of counters, not a multiplier.

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


def _distinct_randint(width: int, k: int, n: int, g: torch.Generator) -> torch.Tensor:
    """(width, k) random ids in [0, n), distinct WITHIN each row.

    Distinct slots means no root-to-leaf path can test one feature twice. A repeated test is a
    redundant gate, and it also makes one branch unreachable, so the invariant costs nothing and
    is worth holding from the very first step. k << n, so rejection converges in a few rounds.
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
    """`width` complete depth-D trees, each reading all `n_in` bits of the previous layer.

    Both tensors are BUFFERS, not parameters -- there is nothing here for an optimizer to
    differentiate:

      feat (width, 2^D - 1)  split feature per internal slot, HEAP ORDER: level d owns slots
                             2^d - 1 .. 2^(d+1) - 2, and cell c of level d is slot 2^d - 1 + c.
      leaf (width, 2^D)      the leaf BIT itself. leaf l is reached by the path whose level-d bit
                             is (l >> (D-1-d)) & 1, i.e. cell = 2*cell + bit descending.
    """

    def __init__(self, n_in: int, width: int, depth: int, g: torch.Generator) -> None:
        super().__init__()
        self.n_in, self.width, self.depth = n_in, width, depth
        self.n_slots = (1 << depth) - 1
        self.n_leaf = 1 << depth
        self.register_buffer("feat", _distinct_randint(width, self.n_slots, n_in, g))
        self.register_buffer("leaf", (torch.rand(width, self.n_leaf, generator=g) > 0.5).float())

    def path_repeats(self) -> int:
        """How many (node, leaf) paths test some feature twice. Must always be 0."""
        bad = 0
        for l in range(self.n_leaf):
            seen, cell = [], 0
            for d in range(self.depth):
                seen.append(self.feat[:, ((1 << d) - 1) + cell])
                cell = cell * 2 + ((l >> (self.depth - 1 - d)) & 1)
            p = torch.stack(seen, 1)
            bad += int(((p[:, :, None] == p[:, None, :]).sum(-1) > 1).any(1).sum())
        return bad

    def cells(self, x: torch.Tensor, feat: torch.Tensor, upto: int) -> torch.Tensor:
        """(B, n_in) -> (B, len(feat)) cell index in [0, 2^upto), routing with the GIVEN feat.

        Sized from `feat`, so the update can route a candidate wiring for a SUBSET of nodes
        without building a second module.
        """
        B, M = x.shape[0], feat.shape[0]
        base = torch.arange(M, device=x.device) * self.n_slots
        cell = torch.zeros(B, M, dtype=torch.long, device=x.device)
        flat = feat.reshape(-1)
        for d in range(upto):
            fi = flat[base + ((1 << d) - 1) + cell]
            cell = cell * 2 + torch.gather(x, 1, fi).long()
        return cell

    def out_of(self, x: torch.Tensor, feat: torch.Tensor, leaf: torch.Tensor) -> torch.Tensor:
        """The layer's output for an arbitrary (feat, leaf) pair -- exact bits, by routing."""
        cell = self.cells(x, feat, self.depth)
        return leaf.reshape(-1)[torch.arange(feat.shape[0], device=x.device) * self.n_leaf + cell]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_of(x, self.feat, self.leaf)


# ==========================================================================================
# The update: what should this node change?
# ==========================================================================================
def _score(cnt0: torch.Tensor, cnt1: torch.Tensor) -> torch.Tensor:
    """Division-free split score: |det| of the 2x2 (side x target) contingency table.

        |n0_t0 * n1_t1 - n0_t1 * n1_t0|      two multiplies and a subtract

    No ratio, so no division and no float -- the point being that this has to be implementable in
    logic. The obvious alternative, "how many samples does this split get right" (max_t n0_t +
    max_t n1_t), is exactly aligned with the node's 0/1 error and is much worse for it: a split
    that purifies a side without flipping its majority label scores identically to one that does
    nothing, so the surface is piecewise-constant and the argmax breaks ties at random. Same
    reason CART splits on Gini rather than error rate; this keeps that sensitivity in the integers.
    """
    return (cnt0[..., 0, :] * cnt1[..., 1, :] - cnt0[..., 1, :] * cnt1[..., 0, :]).abs()


def leaves_and_err(layer: TreeLayer, X: torch.Tensor, feat: torch.Tensor,
                   tgt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Best leaves for a given wiring, and the disagreement they leave, per node.

    Rules follow wiring: for any structure the best leaf is the majority target in that cell, so a
    candidate rewiring is always scored at its own best rules rather than against stale ones.
    """
    M = feat.shape[0]
    cell = layer.cells(X, feat, layer.depth)
    pos = torch.zeros(M, layer.n_leaf, device=X.device, dtype=X.dtype)
    neg = torch.zeros_like(pos)
    pos.scatter_add_(1, cell.t(), tgt.t().contiguous())
    neg.scatter_add_(1, cell.t(), (1 - tgt).t().contiguous())
    leaf = (pos > neg).to(X.dtype)
    out = leaf.reshape(-1)[torch.arange(M, device=X.device) * layer.n_leaf + cell]
    return leaf, (out != tgt).to(X.dtype).sum(0)


def best_wiring(layer: TreeLayer, X: torch.Tensor, tgt: torch.Tensor, feat: torch.Tensor, *,
                mtry: int, chunk: int, gen: torch.Generator | None) -> torch.Tensor:
    """For every slot, the candidate input bit that best separates the target there.

    Level by level, all nodes and all cells at once. At level d the batch is partitioned into 2^d
    cells per node; scattering a 1 into column (node, cell, target) and multiplying by X counts,
    for every candidate bit, how many samples of each target value have it set. X is binary and
    the scatter matrix is one-hot, so this "GEMM" is a bank of counters -- it is written as a
    matmul because that is how a GPU counts a bitpacked batch quickly, not because anything is
    being multiplied.

    `mtry` restricts a chunk's candidates to a random subset: tractability, and diversity, since a
    deterministic argmax would otherwise hand identical trees to every node fed the same target.
    """
    dev, B, M, D = X.device, X.shape[0], feat.shape[0], layer.depth
    new = torch.zeros_like(feat)
    for d in range(D):
        K = 1 << d
        cell = layer.cells(X, new, d)
        col_base = cell * 2 + tgt.long()
        c_ar = torch.arange(K, device=dev)
        for m0 in range(0, M, chunk):
            m1 = min(m0 + chunk, M)
            Mc = m1 - m0
            if mtry and mtry < layer.n_in:
                cand = torch.randperm(layer.n_in, generator=gen, device=dev)[:mtry]
                Xc = X[:, cand]
            else:
                cand, Xc = torch.arange(layer.n_in, device=dev), X

            col = torch.arange(Mc, device=dev) * (K * 2) + col_base[:, m0:m1]
            G = torch.zeros(B, Mc * K * 2, device=dev, dtype=X.dtype)
            G.scatter_(1, col, torch.ones_like(col, dtype=X.dtype))

            cnt1 = (G.t() @ Xc).reshape(Mc, K, 2, -1)
            cnt0 = G.sum(0).reshape(Mc, K, 2, 1) - cnt1
            score = _score(cnt0, cnt1)
            dead = (cnt0.sum(-2) < 1e-9) | (cnt1.sum(-2) < 1e-9)

            # never re-test a bit this cell's path already decided: one side would be empty, and
            # for an unreachable cell `dead` cannot catch it, so mask ancestors explicitly
            if d:
                anc = torch.stack([new[m0:m1][:, ((1 << dd) - 1) + (c_ar >> (d - dd))]
                                   for dd in range(d)], -1)
                banned = (cand.view(1, 1, -1) == anc.unsqueeze(-1)).any(-2)
                dead = dead | banned
            score = score.masked_fill(dead, float("-inf"))

            best = score.argmax(-1)
            alive = score.gather(-1, best[..., None]).squeeze(-1) > float("-inf")
            fallback = ((~banned).to(torch.uint8).argmax(-1) if d else torch.zeros_like(best))
            new[m0:m1, c_ar + (K - 1)] = cand[torch.where(alive, best, fallback)]
    return new


def update_layer(layer: TreeLayer, X: torch.Tensor, tgt: torch.Tensor, *, topk: int = 1,
                 mtry: int = 0, chunk: int = 256, gen: torch.Generator | None = None) -> float:
    """Change `topk` of each node's 2^D - 1 decisions, toward its target. Returns move rate.

    A node has few enough slots that every single-decision change can be scored EXACTLY -- rewire
    that one slot, refit the leaves it implies, count the disagreement -- rather than ranked by a
    proxy. So `topk` is a genuine step size: k=1 means each node changes the one decision that
    helps it most, which is what keeps a whole layer from lurching the same way at once.

    A node keeps the result only if its own disagreement went down. That test is local: it needs
    the node's error bits and its own inputs, nothing from any other node.
    """
    if mtry and mtry <= layer.depth:
        raise ValueError(f"mtry={mtry} must exceed depth {layer.depth}")
    old = layer.feat
    cand = best_wiring(layer, X, tgt, old, mtry=mtry, chunk=chunk, gen=gen)
    _, err_old = leaves_and_err(layer, X, old, tgt)

    if topk and topk < layer.n_slots:
        gain = torch.empty(layer.width, layer.n_slots, device=X.device, dtype=X.dtype)
        for s in range(layer.n_slots):
            trial = old.clone()
            trial[:, s] = cand[:, s]
            _, e = leaves_and_err(layer, X, trial, tgt)
            gain[:, s] = err_old - e
        take = gain.topk(topk, dim=1).indices
        merged = old.clone()
        merged.scatter_(1, take, cand.gather(1, take))
        cand = merged

    leaf, err_new = leaves_and_err(layer, X, cand, tgt)
    keep = err_new < err_old
    layer.feat[keep] = cand[keep]
    layer.leaf[keep] = leaf[keep]
    return float(keep.float().mean())


# ==========================================================================================
# The network
# ==========================================================================================
class TaoNet(torch.nn.Module):
    """Encoder + tree layers + the benchmark's popcount/argmax readout.

    The readout is hw.emit_popcount_argmax's function -- contiguous equal groups, popcount, argmax
    with ties to the lowest class -- identical to backprop/dfa/hebbian, so the comparison across
    records is at matched readout. It is also the only place anything is summed across a group;
    every other count in this file is local to one node.
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
        """(N, 784) uint8 -> (N, 784*bits) bits, laid out exactly as hw.emit_thermometer."""
        t = torch.tensor(self.thresholds, device=pix.device, dtype=torch.int16)
        return (pix.to(torch.int16).unsqueeze(-1) > t).reshape(pix.shape[0], -1).float()

    def activations(self, pix: torch.Tensor) -> list[torch.Tensor]:
        acts = [self.encode(pix)]
        for lay in self.layers:
            acts.append(lay(acts[-1]))
        return acts

    def group_votes(self, last: torch.Tensor) -> torch.Tensor:
        g = self.widths[-1] // N_CLASSES
        return last.reshape(last.shape[0], N_CLASSES, g).sum(-1)

    def forward(self, pix: torch.Tensor) -> torch.Tensor:
        return self.group_votes(self.activations(pix)[-1])

    def predict(self, pix: torch.Tensor) -> torch.Tensor:
        return self.forward(pix).argmax(1)          # ties -> lowest class, as the argmax emits

    def scores(self, pix: torch.Tensor) -> torch.Tensor:
        return self.group_votes(self.activations(pix)[-1]) / (self.widths[-1] // N_CLASSES)


# ==========================================================================================
# The backward pass: one bit per node, and where it comes from
# ==========================================================================================
@torch.no_grad()
def flip_votes(layer: TreeLayer, x: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """(B, n_in) signed demand this layer places on each of its input bits.

    For each bit it reads, a node asks: would flipping you fix me? Exactly answerable -- flip it,
    fall into the sibling subtree, route the rest of the way with the real bits, read the leaf:

        vote(m -> f) = [output with f flipped hits t_m] - [output now hits t_m]

    +1 if the flip would fix node m, -1 if it would break a node that is currently right, 0 if it
    changes nothing. The negative votes matter as much as the positive ones: without them every
    bit anyone wants flipped gets flipped, and the net oscillates. Only bits on the path taken can
    score anything -- off the path, flipping changes nothing -- so the message is sparse for free.
    """
    dev, B, M = x.device, x.shape[0], layer.width
    feat, leaf = layer.feat, layer.leaf
    bs = torch.arange(M, device=dev) * layer.n_slots
    bl = torch.arange(M, device=dev) * layer.n_leaf
    flat = feat.reshape(-1)

    cells, feats, bits = [], [], []
    cell = torch.zeros(B, M, dtype=torch.long, device=dev)
    for d in range(layer.depth):
        fi = flat[bs + ((1 << d) - 1) + cell]
        bit = torch.gather(x, 1, fi).long()
        cells.append(cell)
        feats.append(fi)
        bits.append(bit)
        cell = cell * 2 + bit
    hit = (leaf.reshape(-1)[bl + cell] == tgt).to(x.dtype)

    votes = torch.zeros_like(x)
    for d in range(layer.depth):
        lc = cells[d] * 2 + (1 - bits[d])                 # the sibling
        for dd in range(d + 1, layer.depth):              # route on as usual
            lc = lc * 2 + torch.gather(x, 1, flat[bs + ((1 << dd) - 1) + lc]).long()
        alt = leaf.reshape(-1)[bl + lc]
        votes.scatter_add_(1, feats[d], (alt == tgt).to(x.dtype) - hit)
    return votes


@torch.no_grad()
def targets(net: TaoNet, x: torch.Tensor, y: torch.Tensor):
    """Yield (layer index, that layer's inputs, its target bits), TOP-DOWN.

    The readout target is read straight off the label -- a node in class c's group should fire
    exactly when c is the answer -- and every layer below gets its target from the votes the layer
    above casts on its bits: wanted flipped on balance means flip. Targets are BITS. The sum over
    a bit's consumers is a counter living at the node that drives it, which is the only place it
    could live in hardware.

    This is a generator because the caller rewires each layer while it is suspended here. That
    ordering is what keeps the pass self-consistent: layer li's input comes from the layers BELOW
    it, which have not been touched yet, so its target stays valid however far it moves. Handing
    every layer a target up front and rewiring bottom-up instead fits each layer against an input
    its predecessor has already destroyed -- survivable when a few nodes move, fatal when they all
    do (it collapses to chance).
    """
    acts = net.activations(x)
    g = net.widths[-1] // N_CLASSES
    grp = torch.arange(net.widths[-1], device=x.device) // g
    tgt = (grp[None, :] == y[:, None]).to(acts[0].dtype)

    for li in range(len(net.layers) - 1, -1, -1):
        X = acts[li]
        yield li, X, tgt
        if li:
            votes = flip_votes(net.layers[li], X, tgt)
            tgt = torch.where(votes > 0, 1 - X, X)


# ==========================================================================================
# Training
# ==========================================================================================
def _dichotomy_targets(y: torch.Tensor, width: int, g: torch.Generator) -> torch.Tensor:
    """(B, width) bits. Node m is fit to a random class dichotomy -- an error-correcting output
    code. The candidate search is deterministic, so without this every node in a layer would build
    the same tree. Codes are redrawn per layer."""
    code = torch.randint(2, (width, N_CLASSES), generator=g, device=y.device)
    while True:
        bad = (code.sum(1) == 0) | (code.sum(1) == N_CLASSES)
        if not bool(bad.any()):
            break
        code[bad] = torch.randint(2, (int(bad.sum()), N_CLASSES), generator=g, device=y.device)
    return code[:, y].t().float()


@torch.no_grad()
def fit(net: TaoNet, data: Mnist, *, device: str = "cpu", seed: int = 0, epochs: int = 60,
        steps: int = 20, rows: int = 512, topk: int = 1, mtry: int = 1024, chunk: int = 256,
        patience: int = 20, log_every: int = 1) -> float:
    """Train. Returns the best val accuracy, leaving `net` holding that state.

    Each step: route a batch, carry one error bit per node back down, let every node change `topk`
    of its decisions. No epochs of gradient descent in between, because there is no gradient.
    """
    gen = torch.Generator(device=device).manual_seed(seed + 1)
    x = torch.from_numpy(np.ascontiguousarray(data.train_x)).to(device)
    y = torch.from_numpy(data.train_y).to(device)
    vx = torch.from_numpy(np.ascontiguousarray(data.val_x)).to(device)
    vy = torch.from_numpy(data.val_y).to(device)

    # give every layer an informative starting tree, bottom-up against random dichotomies
    idx = torch.randperm(x.shape[0], generator=gen, device=device)[:2048]
    h = net.encode(x[idx])
    for li, lay in enumerate(net.layers):
        t = _dichotomy_targets(y[idx], lay.width, gen)
        feat = best_wiring(lay, h, t, lay.feat, mtry=mtry, chunk=chunk, gen=gen)
        lay.feat.copy_(feat)
        lay.leaf.copy_(leaves_and_err(lay, h, feat, t)[0])
        print(f"  [init] layer {li}: {lay.width} nodes, dichotomy agreement "
              f"{100 * float((lay(h) == t).float().mean()):.1f}%", flush=True)
        h = lay(h)

    def val_acc() -> float:
        ok = sum(int((net.predict(vx[i:i + 1024]) == vy[i:i + 1024]).sum())
                 for i in range(0, vx.shape[0], 1024))
        return 100.0 * ok / vx.shape[0]

    best, best_ep = val_acc(), -1
    best_state = {k: v.clone() for k, v in net.state_dict().items()}
    print(f"[init] val {best:.2f}%", flush=True)

    t0 = time.time()
    for ep in range(epochs):
        moved = []
        for _ in range(steps):
            idx = torch.randperm(x.shape[0], generator=gen, device=device)[:rows]
            rx, ry = x[idx], y[idx]
            for li, X, tgt in targets(net, rx, ry):
                moved.append(update_layer(net.layers[li], X, tgt, topk=topk, mtry=mtry,
                                          chunk=chunk, gen=gen))

        acc = val_acc()
        if acc > best:
            best, best_ep = acc, ep
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        if ep % log_every == 0 or ep == epochs - 1:
            print(f"  epoch {ep + 1:3d}/{epochs}  val {acc:.2f}%  (best {best:.2f}% @ "
                  f"{best_ep + 1})  moved {100 * float(np.mean(moved)):.0f}%  "
                  f"{time.time() - t0:.0f}s", flush=True)
        if ep - best_ep >= patience:
            print(f"  early stop at epoch {ep + 1}: no gain since {best_ep + 1}", flush=True)
            break

    net.load_state_dict(best_state)
    return best


# ==========================================================================================
# The reference router: pure numpy. The torch path must reproduce it BIT FOR BIT -- the local
# stand-in for the harness's predict()-vs-netlist check.
# ==========================================================================================
def route_numpy(x: np.ndarray, feat: np.ndarray, leaf: np.ndarray) -> np.ndarray:
    B, M, S, L = x.shape[0], feat.shape[0], feat.shape[1], leaf.shape[1]
    base = np.arange(M) * S
    cell = np.zeros((B, M), np.int64)
    flat = feat.reshape(-1)
    for d in range(int(math.log2(L))):
        fi = flat[base + ((1 << d) - 1) + cell]
        cell = cell * 2 + x[np.arange(B)[:, None], fi]
    return leaf.reshape(-1)[np.arange(M) * L + cell].astype(np.uint8)


def predict_numpy(net: TaoNet, pix: np.ndarray) -> np.ndarray:
    thr = np.asarray(net.thresholds, np.int16)
    h = (pix.astype(np.int16)[:, :, None] > thr).reshape(len(pix), -1).astype(np.uint8)
    for lay in net.layers:
        h = route_numpy(h, lay.feat.cpu().numpy(), lay.leaf.cpu().numpy().astype(np.uint8))
    g = net.widths[-1] // N_CLASSES
    return h.reshape(len(pix), N_CLASSES, g).sum(-1).argmax(1)


# ==========================================================================================
# Area estimate. Pre-ABC: it prices one node in isolation and cannot see the sharing ABC finds
# between them. Order of magnitude, not a leaderboard number -- only yosys produces those.
# Constants calibrated against the two measured backprop points that share this encoder and
# readout (xs and s, both bits=1, whose encoder is free, which isolates the head).
# ==========================================================================================
AND_GE = 1.5
MUX_GE = 3.0
READOUT_GE = 5.7
FIXED_GE = 50.0
_CONST, _WIRE, _GATE = 0, 1, 2


def thresh_ge(t: int) -> float:
    """Area of `pix > t` in GE. Not a generic comparator: it only has to look at the bits above
    the trailing run of ones in t+1. hw.even_thresholds lands on those boundaries, so `pix > 127`
    is bit 7 -- a wire -- and `pix > 63` is one gate."""
    c = int(t) + 1
    return max(0, 7 - ((c & -c).bit_length() - 1)) * AND_GE


def _prune_node(feat: np.ndarray, leaf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bottom-up cost per node and which slots survive. A subtree whose leaves are all equal is a
    constant and costs nothing; opposite constants collapse to the literal; one constant child is
    an AND/OR; otherwise a mux."""
    M, L = leaf.shape
    kind = np.full((M, L), _CONST, np.int8)
    val = leaf.astype(np.int8)
    cost = np.zeros((M, L), np.float64)
    live = np.zeros_like(feat, bool)
    for d in range(int(math.log2(L)) - 1, -1, -1):
        K = 1 << d
        kl, kr = kind[:, 0::2], kind[:, 1::2]
        vl, vr = val[:, 0::2], val[:, 1::2]
        cl, cr = cost[:, 0::2], cost[:, 1::2]
        both = (kl == _CONST) & (kr == _CONST)
        same = both & (vl == vr)
        flip = both & (vl != vr)
        one = (kl == _CONST) ^ (kr == _CONST)
        kind = np.where(same, _CONST, np.where(flip, _WIRE, _GATE)).astype(np.int8)
        cost = np.where(same, 0.0, np.where(flip, 0.0,
                        np.where(one, cl + cr + AND_GE, cl + cr + MUX_GE)))
        val = np.where(same, vl, 0).astype(np.int8)
        live[:, K - 1:2 * K - 1] = ~same
    return cost[:, 0], live


def estimate_gates(net: TaoNet) -> dict:
    total, live_slots, per_layer, used = 0.0, 0, [], set()
    for li, lay in enumerate(net.layers):
        feat = lay.feat.cpu().numpy()
        cost, live = _prune_node(feat, lay.leaf.cpu().numpy())
        if li == 0:
            used = set(feat[live].tolist())
        per_layer.append({"width": lay.width, "ge": round(float(cost.sum())),
                          "live_slots": int(live.sum()), "free_nodes": int((cost == 0).sum())})
        total += float(cost.sum())
        live_slots += int(live.sum())
    enc = sum(thresh_ge(net.thresholds[f % net.bits]) for f in used)
    head = READOUT_GE * net.widths[-1] + FIXED_GE
    return {"ge_est": round(total + enc + head), "logic": round(total), "encoder": round(enc),
            "head": round(head), "live_slots": live_slots, "enc_bits_used": len(used),
            "layers": per_layer}
