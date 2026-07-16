# sbuehrer/forest

Boost decision trees on the thermometer bits; ship the ensemble as gates.

A decision tree over binary features **is** a boolean function. Every root-to-leaf path is a
conjunction of literals, so a tree's leaf indicators are a shared-prefix AND network, and the
paths reaching a class form a DNF. Boost a forest, weight each tree by an integer, sum per class,
argmax. No gradient, no LUT net, no search — and the whole model is already a circuit.

```
each round:
    build a tree on the CURRENT MISTAKES (leaf-wise best-first, weighted Gini)
    alpha = (ln((1-e)/e) + ln(K-1)) * lr,  quantized to an integer NOW, not later
    up-weight every misclassification using the QUANTIZED alpha
```

Over binary features a split search is one GEMM — `wyoh.t() @ X` counts, per class, how many
weighted samples have each bit set — so a forest trains in **seconds**, not the hours the
gradient records need.

## What actually makes it cheap

**SAMME, not gradient boosting — but not for the obvious reason.** The tempting argument is that
GBDT puts a real-valued 10-vector on every leaf, so its head sums `T*L` terms per class against
SAMME's `T`. That is only true of a naive emission. Exactly one leaf per tree is hot, so the
one-hot collapse flattens GBDT's leaf dimension exactly as it flattens SAMME's; both heads end up
~`B` adder bits per tree per class. **The adders are a wash.** SAMME wins on two other things:

* **The class indicator partitions.** Each leaf carries one class, so the ten per-class ORs are
  disjoint (~`L` per tree). A per-leaf weight vector needs, for every (class, weight-bit) pair, an
  OR over an arbitrary ~`L/2` subset of leaves — `10*B` overlapping OR-trees, ~15x more select
  logic at `B=3`.
* **One trie feeds ten classes.** Multiclass GBDT grows one tree *per class* per round: 10x the
  trie logic for the same number of rounds.

**Buy conjunctions, not arithmetic.** ABC never instantiates sky130's `fa_1` cell (5.33 GE) — it
builds a full adder from `2x xor2 + maj3` (~7.3 GE), and a measured popcount+argmax head costs
**~9.7 GE per vote bit**, more than the d=8 conjunction it is summing (~5 GE). The sum, not the
DNF, is the architecture.

**Leaf indicators as `reach` wires.** `reach(child) = reach(parent) & ±literal` is exactly 2 gates
per internal node, so **a tree costs `2(L-1)` gates regardless of its shape**. Area tracks the
leaf count, not the depth — which is why `max_depth` is non-binding here: capping depth would
remove capacity at zero area saving. There is no width/depth tradeoff to tune.

## The two things measurement overturned

**Free encoder, false economy.** `pix > 127` is bit 7 of the byte — a wire, zero gates — so the
obvious play is to binarize free and spend everything on trees. Wrong:

| thermometer bits | val @ 40x128 |
|---|---|
| 1 (127) | 94.97% |
| 3 (63,127,191) | 95.90% |

Trees dominate the area, so the encoder is a few percent of the circuit. Every `2^k-1` threshold
is a compare of the top bits only, and only the (pixel, threshold) pairs some node splits on get
emitted — a 5-tree forest touches **33 of 2,352**. Resolution is cheap, and it pays most where
leaves are scarce (+1.3 points at ~400 leaves, +0.15 at ~5,000): with few splits, each must carry
more information. Three of the seven points are `bits=7`; none are `bits=1`.

**Many weak learners — until you count gates.** Boosting theory says spend the budget on many
small trees, and a leaf-count grid agrees loudly: at ~430 leaves, `27x16` beats `3x128` by **7.7
points**. But leaves are not the axis. Vote bits scale with `T` while leaves scale with `T*L`, so
many tiny trees drown in head. At **matched silicon** the ranking inverts:

| budget | L=16 | L=32 | L=64 | L=128 | L=256 | L=384 |
|---|---|---|---|---|---|---|
| ~4k GE | 90.08 | 90.62 | **91.21** | | | |
| ~21k GE | | 94.40 | 95.97 | **96.21** | | |
| ~33k GE | | | | **97.06** | 96.82 | 96.55 |

An **interior optimum that drifts right with the budget** — neither extreme, and invisible in leaf
space. The per-leaf price says why: `L=128` costs ~4.7 GE/leaf, `L=16` costs ~8.8–10.5, because
the head amortizes over 8x fewer leaves.

## Points

Not hand-picked. A 72-config grid over `leaves x wbits x bits` was swept — the tree-count axis
comes free, since the first `t` trees *are* the round-`t` ensemble — then **39 candidates were
synthesized** and these seven are what survived on real silicon, thinned to ~1.8x steps.

| point | bits | leaves | wbits | trees |
|---|---|---|---|---|
| xxs | 3 | 8 | 2 | 5 |
| xs | 7 | 16 | 2 | 9 |
| s | 3 | 128 | 3 | 5 |
| m | 3 | 128 | 3 | 13 |
| l | 7 | 128 | 2 | 30 |
| xl | 3 | 256 | 2 | 39 |
| xxl | 7 | 256 | 2 | 78 |

```bash
python -m mnistbench run records/sbuehrer/forest --device cuda
```

## Caveats

**Val selection bias.** The grid picks the max over 72 configs x every prefix on a 6,000-image val
set, so the winner's *val* number is optimistically biased. This cannot inflate the board: the
harness reports test accuracy simulated from the synthesized netlist, so the bias shows up
honestly as a val−test gap. Empirically test lands *above* val at every point (97.31 vs 97.28 at
`xxl`) — the same direction the genetic record shows, i.e. a property of this fixed split rather
than of the selection.

**Weights are barely weighted.** With `wbits=3` the learned alphas collapse to 5–7, and at `xl`
62 of 64 trees get the same weight. The bit-plane head is real but it is carrying little
information; `wbits=1` (a pure majority vote) is within noise at several budgets. The title says
"integer-weighted vote" and the integers are nearly all equal.

## Prior art

**TreeLUT** (FPGA'25, [arXiv:2501.01511](https://arxiv.org/abs/2501.01511)) is the closest
published match — comparators → per-tree DNF over single-bit keys → quantized adder tree, LUTs
only. MNIST 96.6% at 4,478 LUT6, and it independently landed on a 3-bit tree weight. **conifer**
(JINST 15 P05026) gives the analogous closed-form FPGA law, `LUTs = 22*n_e + 53*n_e*2^d` — linear
in ensemble size, exponential in depth, the same shape as the cost model here. Neither reports
gate equivalents; no decision-tree paper does, which is why this record's axis is worth having.

## Where this line runs out

`xxl` reaches 97.31%, but XGBoost at depth 4 reaches ~97.75% on binarized MNIST, so the tree
ensemble itself is near its ceiling here — and an unbounded random forest needs **596,307 leaves**
to reach 96.78%, which `xxl` beats with 36x fewer. The real headroom is elsewhere: on identical
binarized MNIST, a decision tree gets 87.8% where a **Tsetlin machine gets 98.2%** (Granmo 2018,
Table 16), and Kégl & Busa-Fekete (ICML'09) found *products of stumps* beat trees (1.26% vs 1.53%
error) — a product of `m` stumps being exactly a conjunction of `m` literals. A tree is a cheap
but weak way to build a DNF: it forces one shared prefix per clause. **Learning in clause space
rather than tree space is the next record**, and it reuses this head verbatim.
