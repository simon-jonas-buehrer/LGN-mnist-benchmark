# tao — a deep network of decision trees, trained by gradient-targeted refitting

**Status: phase-1 prototype. Not a submission yet.** There is no `submission.py`, no Verilog and
no measured point on the leaderboard. This directory answers one question first — *does it
learn?* — because the architecture is worth building only if it does.

## The idea

`forest` owns the frontier but is shallow: every tree reads the raw thermometer bits, so nothing
composes. `backprop` is deep but its wiring search is crippled — each gate input picks among **8
randomly drawn candidate signals**, out of thousands.

So: a layered network where every node is a small decision tree with a **full receptive field over
the previous layer**, and the wiring is chosen by information gain over *all* candidate bits.

```
thermometer encoder -> M0 tree-nodes -> M1 tree-nodes -> ... -> popcount -> argmax
```

The tree builder *is* the wiring optimizer. That is the claim, and it is a controlled comparison
against `backprop` — same depth, same encoder, same readout, same exact-hard binary forward, and
the only difference is what chooses the wires.

## Why gradients and trees compose

**The gradient w.r.t. an input bit is exact.** A tree's output is multilinear in the bits it
reads, `out(x) = Σ_l v_l · Π_path lit(x_f)`. On binary inputs the derivative of a multilinear
function is a finite difference with no truncation error at all:

```
d out / d x_f  =  out(x_f = 1) - out(x_f = 0)
flipping bit f moves the output by exactly (1 - 2*x_f) * d out / d x_f
```

Every off-path product contains a zero factor, so **only the ≤D features on the path actually
taken receive gradient**. The "send gradient only to the inputs this node's tree used" property is
not engineered — it falls out of the algebra. `backprop`'s `LutLayer` already uses the 2-input
case of the same form.

That identity has one precondition, which `--gradcheck` found: **no root-to-leaf path may test the
same feature twice.** A repeat puts `x_f·x_f` in the product, and `x² = x` is true as a function on
{0,1} but false as a derivative — a contradictory repeat is identically zero yet carries gradient
`1 − 2x_f`, pointing somewhere the output cannot go. The invariant is enforced structurally in both
the init and the refit. It costs nothing: re-testing a feature the path already decided is a
redundant gate anyway.

**Gradient and refit are one update, split by what each can move.** Gradient moves leaf values
(continuous latents, straight-through). It cannot move a split feature — that is a discrete jump.
So the loop alternates: gradient tunes leaves, and a periodic greedy refit moves wires, fitting
each node to a target read off its own gradient,

```
target[b, m] = 1[g[b, m] < 0]     which way should this bit have gone
weight[b, m] = |g[b, m]|          how much did it matter
```

a weighted binary classification problem per node — exactly what `forest`'s builder solves. Over
binary features a split search is a GEMM, and here it is batched over every node and every cell of
a layer at once instead of looped.

## What phase 1 found

**The refit needs a trust region, and an honest acceptance test.** The first working version
rebuilt every node whose local target error improved. It was actively destructive: gradient alone
reached 79% in one epoch, and every refit round cratered it to ~40%, with the next epoch clawing
back to ~70%. Two causes, both real:

- the target is a *linearisation* of the loss around the current bits, so a tree that perfectly
  matches `sign(−g)` is an enormous jump into territory where `g` means nothing;
- rebuilding a node resets its leaves to the target's weighted majority, discarding what the
  gradient had learned — and ~65% of nodes were being rebuilt every round.

Both are fixed by making the structural step small and checking it against the thing actually being
minimised: each round rebuilds only the worst `refit_frac` of nodes (default 10%), one layer at a
time, and a layer is **reverted outright if the true loss on the refit batch got worse**. With
that, loss falls monotonically at every accepted step and the revert fires when it should.

**The alternation earns its complexity.** `--ablate` at a deliberately tiny 416-node net
(`--widths 256,160 --bits 3 --mtry 256`, 40 epochs, one seed):

| | val acc | est. GE |
|---|---|---|
| **full: dichotomy init + gradient + refit** | **85.98%** | ~3,383 |
| gradient only (structure frozen after init) | 80.67% | ~3,386 |
| refit only (no gradient on the leaves) | 83.12% | ~4,475 |

+5.3 points over gradient-only at *identical* area, and +2.9 over refit-only at 25% *less* area —
the refit-only net spends more silicon because nothing ever collapses its trees toward constants.
Both halves are needed, which is the result this prototype existed to establish.

Not yet run: the depth ablation (deep stack vs one flat layer at matched node count). Until that
lands, "composition is what helps" is untested — a flat net of the same trees might do as well, in
which case this is a worse `forest`.

## Running it

```bash
python records/sbuehrer/tao/proto.py --selfcheck     # multilinear forward == tree routing, bit for bit
python records/sbuehrer/tao/proto.py --gradcheck     # grad == finite difference, and only on-path
python records/sbuehrer/tao/proto.py --widths 1024,512,320 --bits 3
python records/sbuehrer/tao/proto.py --ablate        # full vs gradient-only vs refit-only
python records/sbuehrer/tao/proto.py --depth-ablate  # deep stack vs one flat layer
```

`--selfcheck` is the local stand-in for the harness's `predict()`-vs-netlist check: the torch
multilinear forward and a pure-numpy tree router must agree bit for bit at every layer, every
activation exactly in {0.0, 1.0}, and no path repeating a feature.

`estimate_gates()` reports a **pre-ABC** area estimate. It prices each node alone and cannot see
the sharing ABC finds between them, so it is an order of magnitude, not a leaderboard number — only
`yosys` produces those. Its constants are calibrated against the two measured `backprop` points
that share this encoder and readout (`xs` and `s`, both `bits=1`, whose encoder is free, which
isolates the head).

## Files

| | |
|---|---|
| `tao.py` | `TreeLayer`, `TaoNet`, the vectorised refit, `fit()`, the numpy reference router, the area estimate |
| `proto.py` | CLI: train, self-check, gradient-check, ablate |

## Phase 2, if the numbers justify it

`submission.py` with `POINTS` / `build()`, emitting each node as a mux tree with constant-collapsed
leaves. Note that `bench.load_record` imports `submission.py` by path, so a sibling `import tao`
does not resolve — `submission.py` will have to put its own directory on `sys.path` first.

Named after Tree Alternating Optimization, the closest existing method: alternate between fitting a
node's tree and the signal it is fit against.
