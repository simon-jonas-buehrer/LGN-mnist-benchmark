# tao — a deep network of decision trees, each rebuilt from the signal above it

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

Nothing about this needs a gradient in the classical sense. A node gets a signal from the trees
above, passes a signal down to the bits it reads, and rebuilds itself. That is the whole optimizer,
and the default (`--signal flip`) touches no derivative anywhere.

## Why a tree's messages are exact

The reason a discrete message suffices is a property of the model, so it is worth stating first.
A tree's output is multilinear in the bits it reads, `out(x) = Σ_l v_l · Π_path lit(x_f)`, and on
binary inputs the derivative of a multilinear function is a finite difference with no truncation
error at all:

```
d out / d x_f  =  out(x_f = 1) - out(x_f = 0)
flipping bit f moves the output by exactly (1 - 2*x_f) * d out / d x_f
```

Every off-path product contains a zero factor, so **only the ≤D features on the path actually
taken carry any signal at all**. "Send signal only to the inputs this node's tree used" is not
engineered — it falls out of the algebra. `backprop`'s `LutLayer` already uses the 2-input case of
the same form.

So a derivative here is not an approximation of a counterfactual — it *is* one. Which means the
counterfactual can be computed directly, discretely, without ever building a derivative, and that
is what the default optimizer does.

That identity has one precondition, which `--gradcheck` found: **no root-to-leaf path may test the
same feature twice.** A repeat puts `x_f·x_f` in the product, and `x² = x` is true as a function on
{0,1} but false as a derivative — a contradictory repeat is identically zero yet carries gradient
`1 − 2x_f`, pointing somewhere the output cannot go. The invariant is enforced structurally in both
the init and the refit. It costs nothing: re-testing a feature the path already decided is a
redundant gate anyway.

## The update rule: messages down, whole trees rebuilt

A node does exactly two things, and both are local. It receives a signal from the trees above
saying what it should have output and how much that mattered; it passes a signal down to the bits
it reads; and then it **redesigns its entire tree** against the signal it got. Not just the leaves
— in binary, a decision *is* the choice of which input bit to test, so changing a decision is
rewiring. Every split feature at every level is re-chosen.

Crucially the downward message does not need a derivative. A node that is not producing its target
asks: *which of the bits I read would fix me?* On a binary tree over binary inputs that has an
**exact** answer — flip the bit, fall into the sibling subtree, route the rest of the way down with
the real bits, read the leaf. So:

```
vote(m -> f) = w_m * ( [output with f flipped hits t_m] - [output now hits t_m] )
```

`+w_m` if flipping `f` would fix node `m`, `−w_m` if it would **break** a node that is currently
right, `0` if it changes nothing. The negative votes matter as much as the positive: without them
every bit anyone wants flipped gets flipped, and the net oscillates. A bit's votes are summed
across all its consumers; wanted-flipped on balance becomes its target, and how loudly it was asked
becomes its weight. That is the same `(target, weight)` pair a gradient would have produced, so the
node's own update is unchanged: a weighted binary classification problem, which is exactly what
`forest`'s builder solves, batched here over every node and cell of a layer into one GEMM per
level.

The readout layer's target comes straight off the label — a node in class `c`'s group should fire
exactly when `c` is the answer. Nothing else needs a loss function.

**The discrete message is exact where a gradient is approximate.** Backprop composes first-order
approximations across layers and through the softmax; this composes exact single-bit counterfactuals.
Its support is identical — the on-path bits whose flip changes the output — so it drops into the
same slot, which is what makes `--signal-ablate` a clean comparison. `signal="flip", do_grad=False`
touches no derivative anywhere.

The optional gradient path (`--signal grad`, `--do-grad`) remains, because leaf values *are*
continuous latents that a gradient can tune, and it is the baseline the discrete rule has to beat.

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

**The discrete message learns, but does not yet match a gradient.** `--signal-ablate`, same net,
60 epochs, one seed:

| arm | val acc | est. GE | wall | best epoch |
|---|---|---|---|---|
| grad signal + leaf gradient | **86.92%** | ~3,214 | 277s | 44/60 |
| flip signal + leaf gradient | 81.82% | ~3,416 | 126s | 9/60 |
| flip only — no derivative anywhere | 78.45% | ~4,259 | **33s** | 57/60 |
| flip only, fully local (no revert) | 76.35% | ~4,220 | 32s | 42/60 |

Worse on both axes, by 5–8 points. The discrete rule is ~8x cheaper per epoch, so equal epochs is
not equal compute — but given 400 epochs it early-stopped at 94 with 76.80%, so the gap is real
and not just a budget artifact. Three attempts to close it, none of which worked:

- *Hinge weighting at the readout.* Uniform `w = 1` lets confidently-correct samples shout as
  loudly as wrong ones, where `|dL/d.|` would have silenced them. Weighting each sample's demand
  by `max(0, 1 - (true votes - best rival votes))` — integer, derivative-free — improved the early
  curve (73.8% vs 71.2% at epoch 6) but not the converged number.
- *Edit-rate step size* (`--topk`): a node computes its best config but applies only its `k` most
  valuable rewirings, each scored exactly (a node has only `2^D - 1` slots, so every single-slot
  change can be evaluated at its own best leaves rather than ranked by a proxy). `topk=1` reached
  67.23% and plateaued — slower, not steadier.
- *No damping at all* — every node rebuilding every batch, on the theory that it is noisy early and
  settles once trees are good enough. It does not settle: it oscillates in a 33–48% band. It is at
  least no longer catastrophic; see the update-order fix below.

**Update order is load-bearing.** Handing every layer its target up front and then rewiring
bottom-up fits each layer against an input its predecessor has already destroyed. Harmless when
10% of nodes move, fatal when all of them do — that configuration collapsed to chance (~7%).
Updating **top-down** is self-consistent, because a layer's input comes from the layers below it,
which have not moved yet; the same run then reaches 48%. `_flip_targets` is a generator for this
reason: the caller rewires a layer while the pass is suspended, so the message sent downward is
cast by the layer as it now stands.

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

**Depth does not yet pay for itself in accuracy.** `--depth-ablate` at matched *node count*:

| | val acc | est. GE |
|---|---|---|
| deep: 256,160 | 85.98% | ~3,383 |
| flat: 410 | **87.40%** | ~5,418 |

The flat net is *more accurate* and the deep net is *cheaper* — mostly because a narrower last
layer buys a smaller readout, and because fewer nodes read the encoder directly. Matched node
count is the wrong control for a benchmark whose axis is area; this has to be rerun at **matched
GE** before either "composition helps" or "composition does not help" can be claimed. As it stands,
nothing here demonstrates that depth is doing the work, which is the central premise.

## Running it

```bash
python records/sbuehrer/tao/proto.py --selfcheck      # multilinear forward == tree routing, bit for bit
python records/sbuehrer/tao/proto.py --gradcheck      # signal == finite difference, and only on-path
python records/sbuehrer/tao/proto.py --widths 1024,512,320 --bits 3   # discrete signal, the default
python records/sbuehrer/tao/proto.py --signal-ablate  # discrete messages vs a real gradient
python records/sbuehrer/tao/proto.py --ablate         # refit vs leaf-gradient vs both
python records/sbuehrer/tao/proto.py --depth-ablate   # deep stack vs one flat layer
```

Knobs that decide what the optimizer is allowed to use: `--signal flip|grad` (discrete
counterfactual votes, or a backward pass), `--do-grad` (also run Adam on the leaf latents), and
`--no-revert` (drop the loss-gated layer revert, the one global check, making the update purely
local).

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
