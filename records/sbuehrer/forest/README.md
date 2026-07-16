# sbuehrer/forest

Boost decision trees on the thermometer bits; ship the ensemble as gates.

## Architecture

A decision tree over binary features **is** a boolean function. Every root-to-leaf path is a
conjunction of literals, so a tree's leaf indicators are a shared-prefix AND network, and the paths
reaching a class form a DNF. Weight each tree by an integer, sum per class, argmax. No gradient, no
LUT net, no search, and the whole model is already a circuit.

Three things make it cheap in silicon:

* **Leaf indicators as `reach` wires**, not flat ANDs. `reach(child) = reach(parent) & ±literal` is
  exactly 2 gates per internal node, so **a tree costs `2(L-1)` gates regardless of its shape**.
  Area tracks the leaf count, not the depth, which is why `max_depth` is non-binding here, and why
  there is no width/depth tradeoff to tune.
* **The class indicator partitions.** Each leaf carries exactly one class, so the ten per-class ORs
  are disjoint (~`L` per tree in total), and a single reach network scores all ten classes.
* **Bit-plane popcount.** `w_t` is a constant and `v[t][c]` is one bit, so
  `score_c = sum_b 2^b * popcount({v[t][c] : bit b of w_t set})`. A zero weight-bit costs no
  hardware at all, halving the adder inputs versus a Wallace tree over `T` `B`-bit numbers.

The encoder is the harness's own thermometer, so `bits` means what it means in the other records.
Thermometer bits rather than the raw pixel bits, even though the raw bits are free wires and
strictly more expressive: greedy Gini cannot use a bit like `pix[6]`, which is non-monotone in
intensity and has ~zero standalone information gain, so it is never selected. Thermometer bits are
monotone and therefore individually informative, which is what a greedy builder needs.

## Optimizer

SAMME (multiclass AdaBoost). The tree builder is leaf-wise best-first on weighted Gini.

```
each round:
    build a tree on the CURRENT MISTAKES (leaf-wise best-first, weighted Gini)
    alpha = (ln((1-e)/e) + ln(K-1)) * lr,  quantized to an integer NOW, not later
    up-weight every misclassification using the QUANTIZED alpha
```

Over binary features a split search is one GEMM (`wyoh.t() @ X` counts, per class, how many
weighted samples have each bit set), so a forest trains in **seconds** rather than the hours the
gradient records need.

Weights are quantized **inside** the boosting loop, not rounded afterwards: the sample-weight update
consumes the integer alpha the circuit will use, so every later tree is fit against the residual
error of the circuit-exact ensemble and the quantization error is boosted away instead of
accumulating. `wscale` is derived per run from the observed alpha distribution, never hand-tuned.
SAMME drops any tree with `err >= 1-1/K`, so alpha is always ≥ 0: every score is unsigned and there
is no two's complement anywhere.

## Points

Not hand-picked. A 72-config grid over `leaves x wbits x bits` was swept (the tree-count axis comes
free, since the first `t` trees *are* the round-`t` ensemble), then 39 candidates were synthesized
and these seven are what survived on real silicon, thinned to ~1.8x steps. Selection is on val; the
harness reports test from the netlist.

`leaves` is the per-tree leaf budget (the only capacity knob, since depth is free), `wbits` the
integer tree-weight width, `trees` the boosting round count.

| point | bits | leaves | wbits | trees |
|---|---|---|---|---|
| xxs | 3 | 8 | 2 | 5 |
| xs | 7 | 16 | 2 | 9 |
| s | 3 | 128 | 3 | 5 |
| m | 3 | 128 | 3 | 13 |
| l | 7 | 128 | 2 | 30 |
| xl | 3 | 256 | 2 | 39 |
| xxl | 7 | 256 | 2 | 78 |

Two measurements set that grid. The free encoder (`pix > 127`) is a false economy: trees dominate
the area, so resolution is cheap and pays most where leaves are scarce; three of the seven points
are `bits=7` and none are `bits=1`. And boosting theory's "many weak learners" inverts at matched
silicon: vote bits scale with `T` while leaves scale with `T*L`, so many tiny trees drown in head,
and the optimum `L` is interior and drifts right as the budget grows.

## Prior art

**TreeLUT** (FPGA'25, [arXiv:2501.01511](https://arxiv.org/abs/2501.01511)) is the closest published
match: comparators, then a per-tree DNF over single-bit keys, then a quantized adder tree. It
independently landed on a 3-bit tree weight. **conifer** (JINST 15 P05026) gives the analogous
closed-form FPGA law, `LUTs = 22*n_e + 53*n_e*2^d`. Neither reports gate equivalents; no
decision-tree paper does, which is why this record's axis is worth having.

```bash
python -m mnistbench run records/sbuehrer/forest --device cuda
```
