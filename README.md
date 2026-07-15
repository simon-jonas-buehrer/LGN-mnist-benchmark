# mnistbench: benchmark optimizers by the silicon their solutions cost

Optimizers are usually compared on different architectures, so the comparison means little: one
paper counts parameters, another counts gates before synthesis, another counts FLOPs. This repo
fixes one task, one dataset, and one cost axis, and lets any optimizer compete on it.

**You submit a training procedure and a circuit. The harness measures the circuit.**

|  | |
|---|---|
| task | MNIST, fixed 54k / 6k / 10k train / val / test split |
| y-axis | test accuracy, from simulating your synthesized netlist gate by gate |
| x-axis | circuit size in gate equivalents (GE): `yosys` + `ABC` map your Verilog to sky130 cells; GE = area / area of a NAND2 |
| result | two curves against GE: accuracy, and cross-entropy loss |

Everything the model does at inference is inside the circuit and is counted: the binarizer, the
learned logic, the readout, the argmax. No free preprocessing, no free softmax. That lets a LUT
net, a quantized MLP, and a boosted tree land on the same axis.

![accuracy vs gate equivalents](results/pareto_acc.png)
![loss vs gate equivalents](results/pareto_loss.png)

The two reference records cross: below a few thousand gate equivalents the genetic search is ahead,
above it backprop pulls away. Solid lines with markers are measured points; the dashed line extends
a power-law fit past the largest circuit actually synthesized.

Every point is trained to convergence: it early-stops when its own validation accuracy stops
improving, not at a fixed budget.

## Leaderboard

<!-- LEADERBOARD -->
| | record | point | gate equivalents | depth | MNIST test acc | test CE |
|---|---|---|---|---|---|---|
| * | `sbuehrer/backprop` | l | 52,973 | 238 | **95.35%** | 0.152 |
| * | `sbuehrer/backprop` | m | 32,425 | 237 | **93.41%** | 0.213 |
| * | `sbuehrer/backprop` | s | 7,514 | 188 | **87.89%** | 0.390 |
| * | `sbuehrer/genetic` | xs | 1,945 | 129 | **80.38%** | 0.627 |
| * | `sbuehrer/backprop` | xs | 1,913 | 130 | **73.10%** | 0.782 |

`*` = on the Pareto frontier (nothing is both smaller and more accurate). test CE = temperature-calibrated cross-entropy over the circuit's class votes.
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
    def emit_verilog(self) -> str: ...               # the trained model, as module top
    def predict(self, pix): ...                      # numpy in, numpy out; must equal the verilog

def build(**point) -> Submission: return Mine(**point)
```

```bash
python -m mnistbench run records/<you>/<method>   # train, synthesize, simulate, write results.json
python -m mnistbench pareto                        # redraw the curves and the table
```

The harness imports numpy and nothing else, so write your model in PyTorch, JAX, TensorFlow, or
raw bit-twiddling; all it sees is arrays and Verilog. If your model is a fan-in-2 logic net,
`mnistbench/hw.py` emits the Verilog for you.

A point whose `predict()` disagrees with its own circuit on any image is rejected, so the accuracy
on the board is the accuracy of the hardware.

## What's here

```
mnistbench/       the harness
  data.py         MNIST as uint8 numpy, fixed split
  spec.py         the Submission API, the whole contract
  hw.py           verilog emitters: thermometer encoder, fan-in-2 LUT layers, popcount + argmax
  synth.py        yosys + ABC to sky130 area (x-axis) and a NAND netlist
  netlist.py      bit-packed simulator, 64 images per uint64 word (y-axis)
  bench.py        train, emit, synth, simulate, write results.json
  pareto.py       the curves and the leaderboard
  selftest.py     checks emit == synthesize == simulate, bit for bit
records/
  sbuehrer/backprop/   gradients (straight-through sin on the truth tables, softmax on the wires)
  sbuehrer/genetic/    no gradients (fixed NANDs, wiring by mutation hill-climbing)
docs/RULES.md
```

The two records share an encoder, a head, and a gate budget, and differ in one thing: whether the
optimizer has a gradient. Backprop learns what each gate computes and what it reads. The
hill-climber learns wiring alone, by mutation. NAND is functionally complete, so the hill-climber
searches a space that contains every circuit backprop can express; what separates them is the
search, not the architecture.

## Running the scorer

Scoring needs `yosys` (with ABC) and the sky130 liberty, which are not pip-installable:

```bash
conda create -n eda -c conda-forge -c litex-hub yosys open_pdks.sky130a
export MNISTBENCH_YOSYS=.../eda/bin/yosys
export MNISTBENCH_LIBERTY=.../sky130_fd_sc_hd__tt_025C_1v80.lib
python -m mnistbench.selftest     # emit, synthesize, simulate, bit-exact
```
