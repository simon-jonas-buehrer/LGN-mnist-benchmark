# tao — a binary network of decision trees, trained by a local binary error signal

**Status: phase-1 prototype. Not a submission yet.** No `submission.py`, no Verilog, no measured
point. This directory answers *does it learn?* first.

## The idea

`forest` owns the frontier but is shallow — every tree reads the raw thermometer bits, so nothing
composes. `backprop` is deep but its wiring search picks among **8 randomly drawn candidates** per
gate input, out of thousands.

So: a layered net where every node is a small decision tree with a **full receptive field over the
previous layer**, and the wiring is chosen by looking at every candidate bit. The tree builder *is*
the wiring optimizer.

```
thermometer encoder -> M0 tree-nodes -> M1 tree-nodes -> ... -> popcount -> argmax
```

**Nothing here is continuous.** No gradient, no float, no loss surface — the goal is a learning
rule that could run in binary logic on an FPGA, not only a model that ends up as one.

## The rule

| | |
|---|---|
| **forward** | route each sample through each tree — the same work the emitted circuit does |
| **error** | at the readout, a node in class `c`'s group should fire exactly when `c` is the answer. Its error is one bit: `out XOR should` |
| **backward** | a node asks which bit it reads would fix it. **Exact**: flip the bit, fall into the sibling subtree, route on, read the leaf. `vote = [flipped hits target] − [now hits target]` ∈ {+1, 0, −1} |
| **update** | a node compares its error against **all** candidate bits over the batch and changes **one** decision — which input it tests, and hence its rule — or the best `topk` of its `2^D − 1` |

No derivative exists or is needed. A tree's output is multilinear in the bits it reads, so on
binary inputs a derivative *is* a counterfactual — which means the counterfactual can just be
computed. Only bits on the path taken can score anything, so the message is sparse for free.

Negative votes matter as much as positive ones: without them, every bit anyone wants flipped gets
flipped and the net oscillates. Signals on the wire are bits; the sum over a bit's consumers is a
counter living at the node that drives it — the only place it could live in hardware.

**Order is load-bearing.** `targets()` is a generator: the caller rewires each layer while the pass
is suspended. A layer's input comes from the layers *below*, untouched, so its target stays valid
however far it moves. Handing every layer a target up front and rewiring bottom-up instead fits
each layer against an input its predecessor already destroyed — survivable when a few nodes move,
**fatal when they all do** (that collapsed to chance).

## Implementable in logic

| operation | in hardware |
|---|---|
| forward | route trees — muxes. A leaf indicator is a conjunction of literals, so a batch is bitwise AND/OR over 64-sample words (as `mnistbench/netlist.py` already simulates netlists) |
| candidate search | X is binary and the scatter matrix one-hot, so the "GEMM" is a **bank of counters** — no multiplier. It is written as a matmul because that is how a GPU counts a bitpacked batch fast |
| split score | `_score`: a 2×2 determinant — two multiplies, one subtract, **no division** |
| split indices | only a GPU needs an address; on an FPGA a split feature *is* a mux select |
| update | a worker visits a node, reads its error bits and candidate inputs, rewrites one decision |

Division-free cost this, measured before the rewrite: Gini (float) 76.35% → determinant (integer)
62.83% → "how many samples does this split get right" (integer) 46.97%. The last is *exactly*
aligned with the node's 0/1 error, which is what makes it bad — a split that purifies a side
without flipping its majority label scores identically to one that does nothing, so the surface is
piecewise-constant and the argmax breaks ties at random. Same reason CART splits on Gini rather
than error rate. Counter width, separately, is **free**: 8-bit and unbounded were bit-identical.

## Where it stands

`--widths 512,320 --bits 3 --mtry 256`, depth 2, topk 1, 30 epochs, one seed, CPU:

**64.70% val @ ~3,446 estimated GE.** It learns, then plateaus around 63–65%.

Two things that plateau points at, both concrete:

- **Only 12–15% of nodes move per step.** Most nodes propose a change and reject it, so the search
  is finding few improvements — the update rule, not the step size, is the limiter.
- **The readout now dominates the area.** At depth 2 the trees cost ~1,276 GE and the popcount head
  costs ~1,874. Widening the last layer is the most expensive thing you can do; a taller stack with
  a *narrow* final layer is the cheap direction. `--stack-ablate` tests exactly that.

Tree depth is fixed at 2 from here: at matched node count accuracy went 81.05 → 86.12 → 88.40 for
depth 2 → 3 → 4, so +5.1 then +2.3 points, against a steady ~1.9× area each step. Capacity comes
from more nodes instead.

Earlier gradient-based variants reached 86.92%, and were deleted in favour of this one; they are in
git history if a comparison is ever wanted.

## Running it

```bash
python records/sbuehrer/tao/proto.py --selfcheck      # torch routing == numpy routing, bit for bit
python records/sbuehrer/tao/proto.py --flipcheck      # votes are exact counterfactuals, on-path only
python records/sbuehrer/tao/proto.py --widths 512,320 --bits 3
python records/sbuehrer/tao/proto.py --stack-ablate   # taller vs wider at matched node count
```

`--selfcheck` is the local stand-in for the harness's `predict()`-vs-netlist check. `estimate_gates()`
is **pre-ABC**: it prices each node alone and cannot see the sharing ABC finds between them, so it
is an order of magnitude, not a leaderboard number. Its constants are calibrated against the two
measured `backprop` points sharing this encoder and readout.

## Phase 2, if the numbers justify it

`submission.py` with `POINTS`/`build()`, emitting each node as a mux tree with constant-collapsed
leaves. `bench.load_record` imports `submission.py` by path, so a sibling `import tao` will not
resolve — it must put its own directory on `sys.path` first.

Named after Tree Alternating Optimization: alternate between fitting a node's tree and the signal
it is fit against.
