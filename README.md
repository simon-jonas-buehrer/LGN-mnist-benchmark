# mnistbench — benchmark optimizers by the silicon their solutions cost

Optimizers get compared on different architectures, and the comparison never means much: one
paper counts parameters, another counts "gates" (before synthesis), another counts FLOPs. This
repo fixes one task, one dataset and one cost axis, and lets any optimizer compete on it.

**You submit a training procedure and a circuit. We measure the circuit.**

|  | |
|---|---|
| **task** | MNIST, fixed 54k / 6k / 10k train / val / test split |
| **y-axis** | test accuracy, measured by **simulating your synthesized netlist** gate by gate |
| **x-axis** | circuit size in **gate equivalents (GE)**: `yosys`+`ABC` map your Verilog to sky130 standard cells; GE = area / area of a NAND2 |
| **result** | a Pareto curve — for a given amount of silicon, whose optimizer finds the best circuit? |

Everything your model does at inference lives inside the circuit and is counted: the binarizer,
the learned logic, the readout, the argmax. No free preprocessing, no free softmax. That is what
lets a LUT net, a quantized MLP and a boosted tree land on the same axis honestly.

![Pareto curve](results/pareto.png)

The two reference records already make the point, and on the log-log error curve they visibly
cross near **5k gate equivalents**. Below that the **genetic** search is ahead — its NAND-only
circuits map to cheaper cells, and at that size wiring is most of what matters, which is all it
learns. Above it **backprop** pulls away: its error falls as a near-straight power law in gates
down to **3.1%** (96.9% accuracy), while the hill-climber bends and flattens near 13% (87%).
Neither record could have shown that by reporting its own parameter count; it only appears once
both are charged for the same silicon.

Every point here is trained to convergence — each one early-stops when its own validation accuracy
stops improving, not at a fixed budget. That is load-bearing: an earlier version of this benchmark
capped the genetic search at 40k generations and concluded it "plateaus at 81%". It does not. Given
room to converge (the `m` point needs ~800k generations, seven GPU-hours) it reaches 86.5%. A
stopping rule chosen for convenience had been masquerading as a property of the algorithm — exactly
the kind of error a shared, re-runnable cost axis is supposed to catch.

## Leaderboard

<!-- LEADERBOARD -->
| | record | point | gate equivalents | area (um^2) | depth | MNIST test acc |
|---|---|---|---|---|---|---|
| * | `sbuehrer/backprop` | xl | 156,861 | 588,792 | 285 | **96.93%** |
| * | `sbuehrer/backprop` | l | 52,973 | 198,838 | 238 | **95.35%** |
| * | `sbuehrer/backprop` | m | 32,425 | 121,712 | 237 | **93.41%** |
| * | `sbuehrer/backprop` | s | 7,514 | 28,205 | 188 | **87.89%** |
|  | `sbuehrer/genetic` | l | 20,114 | 75,501 | 206 | **87.28%** |
|  | `sbuehrer/genetic` | m | 9,146 | 34,330 | 191 | **86.52%** |
| * | `sbuehrer/genetic` | s | 3,920 | 14,715 | 156 | **83.16%** |
| * | `sbuehrer/genetic` | xs | 1,945 | 7,301 | 129 | **80.38%** |
| * | `sbuehrer/backprop` | xs | 1,913 | 7,179 | 130 | **73.10%** |

`*` = on the Pareto frontier (nothing is both smaller and more accurate).
<!-- /LEADERBOARD -->

## The contract

```verilog
module top (input [6271:0] pix, output [3:0] cls);   // combinational; no clock, no memory
```

`pix[8*p +: 8]` is pixel `p` as a raw uint8 (row-major, `p = 0..783`); `cls` is the predicted
digit. Full rules in [docs/RULES.md](docs/RULES.md).

## Submit

```python
# records/<you>/<method>/submission.py
POINTS = [{"name": "s", ...}, {"name": "l", ...}]    # one dict per point on your curve

class Mine(Submission):
    def train(self, data, *, device, seed): ...      # data.train_x is (54000, 784) uint8 numpy
    def emit_verilog(self) -> str: ...               # the trained model, as `module top`
    def predict(self, pix): ...                      # numpy in, numpy out; must equal the verilog

def build(**point) -> Submission: return Mine(**point)
```

```bash
python -m mnistbench run records/<you>/<method>   # train -> synthesize -> simulate -> results.json
python -m mnistbench pareto                       # redraw the curve and the table
```

The harness imports **numpy and nothing else** — write your model in PyTorch, JAX, TensorFlow or
raw bit-twiddling; all we ever see is arrays and Verilog. If your model is a fan-in-2 logic net
(most are), `mnistbench/hw.py` emits the Verilog for you.

A point whose `predict()` disagrees with its own circuit on even one image is rejected, so the
accuracy on the board is always the accuracy of the hardware.

## What's here

```
mnistbench/       the harness
  data.py         MNIST as uint8 numpy, fixed split
  spec.py         the Submission API — the whole contract
  hw.py           verilog emitters: thermometer encoder, fan-in-2 LUT layers, popcount + argmax
  synth.py        yosys + ABC -> sky130 area (x-axis) and a NAND netlist
  netlist.py      bit-packed simulator, 64 images per uint64 word (y-axis)
  bench.py        train -> emit -> synth -> simulate -> results.json
  pareto.py       the curve and the leaderboard
  selftest.py     proves emit == synthesize == simulate, bit for bit
records/
  sbuehrer/backprop/   gradients (straight-through sin on the truth tables, softmax on the wires)
  sbuehrer/genetic/    no gradients (fixed NANDs, wiring by mutation hill-climbing)
docs/RULES.md
```

The two records share an encoder, a head and a gate budget, and differ in the one thing this
benchmark is about: whether the optimizer has a gradient. Backprop learns both halves of a gate —
what it computes and what it reads. The hill-climber learns wiring alone, by mutation, and NAND
is functionally complete, so it is searching a space that *contains* every circuit backprop can
express. What separates them on the curve is the search, not the hypothesis class. Beat them.

## Running the scorer

Scoring needs `yosys` (with ABC) and the sky130 liberty — not pip-installable:

```bash
conda create -n eda -c conda-forge -c litex-hub yosys open_pdks.sky130a
export MNISTBENCH_YOSYS=.../eda/bin/yosys
export MNISTBENCH_LIBERTY=.../sky130_fd_sc_hd__tt_025C_1v80.lib
python -m mnistbench.selftest     # emit -> synthesize -> simulate, bit-exact
```
