# The rules

You submit a training procedure and a circuit. The harness measures the circuit.

## The axes

| axis | what it is | where it comes from |
|---|---|---|
| x | circuit size, in gate equivalents (GE) | `yosys` + `ABC` map your Verilog to the sky130 standard-cell library; GE = total cell area / area of a sky130 NAND2 (3.7536 um^2) |
| y | MNIST test accuracy | the same Verilog is mapped a second time to a NAND-only netlist, and that netlist is evaluated gate by gate on all 10,000 test images |

There is also a loss curve: if `scores()` returns the readout's per-class vote fractions, the
harness reports a temperature-calibrated cross-entropy. It is optional; accuracy is not.

Nothing you report about your own model is used. `predict()` exists only so the harness can check
that your python model and your Verilog are the same function; if they disagree on even one of 512
sampled images, the point is rejected.

### Why gate equivalents, and not "number of gates"

Every logic-network paper reports a *pre-synthesis architectural* count -- one gate per neuron,
before any optimization -- and every architecture counts a different thing. A fan-in-2 LUT, a
fan-in-6 LUT, a 4-bit adder and an XOR are all "one gate" in somebody's paper, and in silicon
they differ by more than an order of magnitude. GE is the unit ASIC designers actually use: it
prices every cell by its area, relative to the simplest useful gate. It is the only number that
means the same thing for a difflogic net, a quantized MLP and a decision tree.

It also means the logic optimizer is allowed to delete whatever you wasted. Dead gates,
constant-driven logic, a pixel nobody reads, a truth table that collapsed to a wire -- all of it
disappears before you are charged. **You pay for the circuit you need, not the one you wrote.**

## The circuit contract

```verilog
module top (input [6271:0] pix, output [3:0] cls);
```

* `pix[8*p +: 8]` is the uint8 value of pixel `p` (row-major, `p = 0..783`). Raw MNIST bytes.
* `cls` is the predicted digit, `0..9`.
* Purely combinational. No clock, no state, no memory.

Everything between those ports is yours and is counted: the binarizer or thermometer encoder, the
learned logic, the readout head, the argmax. No free preprocessing and no free softmax. That is
what lets a 3-layer LUT net, a boosted tree ensemble and a 2-bit quantized MLP land on one axis:
whatever architecture you bring becomes gates, and the gates are counted the same way.

Cheap design choices are rewarded. `pix > 127` is bit 7 of the byte, a wire, zero gates. `pix >
100` is a real comparator. If your encoder spends 8,000 GE before any learning happens, that
8,000 GE is on your x-axis.

## What you may and may not do

* Train on `data.train_x/train_y` (54,000 images). Tune on `data.val_x/val_y` (6,000).
* **Never fit on the test set.** The harness holds it; you do not need it.
* Any optimizer, any architecture, any random seed, any amount of compute. There is no compute
  budget; the axes are area and accuracy. (`train_s` is recorded in `results.json` as information,
  not as a constraint.)
* Data augmentation is allowed (it changes your optimizer's inputs, not the circuit's).
* One record = one method = one curve. Sweep model sizes to get several points; that is your
  curve, and the union of everyone's curves is the frontier.

## Submitting

1. `mkdir -p records/<you>/<method>` and write `submission.py` with `POINTS` and
   `build(**point) -> Submission` (the API is `mnistbench/spec.py`; the two records under
   `records/sbuehrer/` are working examples).
2. `python -m mnistbench run records/<you>/<method>` -- trains, synthesizes, simulates, writes
   `results.json`. (Budget a few minutes of ABC per point; a 100k-gate circuit takes ~3 min.)
3. `python -m mnistbench pareto` -- rebuilds the plot and the leaderboard.
4. Open a PR with `submission.py`, `results.json` and a short `README.md` saying what your
   optimizer does. (Weights and generated Verilog are not committed; they regenerate.)

`python -m mnistbench rescore records/<you>/<method>` re-measures the stored `.sv` artifacts
without retraining -- both axes come from the Verilog alone, so anyone can re-derive your numbers
from your circuit.

The harness itself imports **numpy and nothing else**: images go in as numpy, predictions come
back as numpy. Write the model in PyTorch, JAX, TensorFlow or none of them -- the benchmark never
sees your framework.

## Reproducing the numbers

Scoring needs `yosys` (with ABC) and the sky130 liberty file -- neither is pip-installable:

```bash
conda create -n eda -c conda-forge -c litex-hub yosys open_pdks.sky130a
export MNISTBENCH_YOSYS=/path/to/eda/bin/yosys
export MNISTBENCH_LIBERTY=/path/to/sky130_fd_sc_hd__tt_025C_1v80.lib
python -m mnistbench.selftest    # proves emit == synthesize == simulate, bit for bit
```

Every submission is synthesized with the *same* ABC script (`mnistbench/synth.py: OPT`), so the
optimization effort is a constant, not something you can tune your way up the leaderboard with.
