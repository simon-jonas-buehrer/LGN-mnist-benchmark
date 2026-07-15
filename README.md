# mnistbench

Compare optimizers by the chip area their solutions cost.

Papers compare optimizers on different models, so the numbers never line up: one counts
parameters, one counts gates, one counts FLOPs. This repo picks one task, one dataset, and one
cost, and lets any optimizer compete.

**You send a training procedure and a circuit. The harness measures the circuit.**

|  | |
|---|---|
| task | MNIST, fixed 54k / 6k / 10k train / val / test split |
| x-axis | circuit size in gate equivalents (GE). `yosys` and `ABC` turn your Verilog into sky130 chip cells; GE = total area / the area of one NAND2 gate. |
| y-axis | test accuracy, and a cross-entropy loss, both read off the built circuit |

Everything the model does at run time is in the circuit and is counted: the input encoding, the
logic, the readout, the argmax. Nothing is free. That is what puts a logic net, a small MLP, and a
boosted tree on the same axis.

![accuracy vs circuit size](results/pareto_acc.png)
![loss vs circuit size](results/pareto_loss.png)

The two example records cross. Below a few thousand gates the genetic search wins; above that,
backprop pulls ahead. Solid dots are measured. The dashed line is a power-law fit, drawn past the
largest circuit we actually built.

Every point trains until it stops improving on the validation set, not to a fixed step count.

## Leaderboard

<!-- LEADERBOARD -->
| | record | point | gate equivalents | depth | MNIST test acc | test CE |
|---|---|---|---|---|---|---|
| * | `sbuehrer/backprop` | xl | 156,861 | 285 | **96.93%** | 0.102 |
| * | `sbuehrer/backprop` | l | 52,973 | 238 | **95.35%** | 0.152 |
| * | `sbuehrer/backprop` | m | 32,425 | 237 | **93.41%** | 0.213 |
| * | `sbuehrer/backprop` | s | 7,514 | 188 | **87.89%** | 0.390 |
| * | `sbuehrer/genetic` | s | 3,920 | 156 | **83.16%** | 0.558 |
| * | `sbuehrer/genetic` | xs | 1,945 | 129 | **80.38%** | 0.627 |
| * | `sbuehrer/backprop` | xs | 1,913 | 130 | **73.10%** | 0.782 |

`*` = on the Pareto frontier (nothing is both smaller and more accurate). test CE = calibrated cross-entropy over the circuit's class votes.
<!-- /LEADERBOARD -->

## Add your optimizer

Write `records/<you>/<method>/submission.py`:

```python
POINTS = [{"name": "s", ...}, {"name": "l", ...}]    # one dict per point on your curve

class Mine(Submission):
    def train(self, data, *, device, seed): ...      # data.train_x is (54000, 784) uint8 numpy
    def emit_verilog(self) -> str: ...               # your trained model, as module top
    def predict(self, pix): ...                      # numpy in, numpy out; must match the verilog

def build(**point) -> Submission: return Mine(**point)
```

Then:

```bash
python -m mnistbench run records/<you>/<method>   # train, build the circuit, measure it
python -m mnistbench pareto                        # redraw the plots and the table
```

The harness only uses numpy, so write the model in PyTorch, JAX, TensorFlow, or plain code. It
only ever sees arrays and Verilog. If your model is a fan-in-2 logic net, `mnistbench/hw.py` writes
the Verilog for you.

Your circuit must match `predict()` on every test image, or the point is dropped. So the score on
the board is the score of the hardware.

The circuit has one fixed shape:

```verilog
module top (input [6271:0] pix, output [3:0] cls);   // combinational; no clock, no memory
```

`pix[8*p +: 8]` is pixel `p` as a raw byte (row-major, `p = 0..783`); `cls` is the digit. Full
rules in [docs/RULES.md](docs/RULES.md).

## What's here

```
mnistbench/     the harness
  data.py       MNIST as uint8 numpy, fixed split
  spec.py       the submission API
  hw.py         Verilog: thermometer encoder, fan-in-2 LUT layers, popcount + argmax
  synth.py      yosys + ABC to chip area and a NAND netlist
  netlist.py    bit-packed simulator, 64 images per word
  bench.py      train, emit, synth, simulate
  pareto.py     the plots and the leaderboard
  selftest.py   checks emit == synth == simulate, bit for bit
records/sbuehrer/
  backprop/     learns each gate and its wiring by gradient descent
  genetic/      fixes every gate to NAND, learns only the wiring by mutation
```

The two records are a matched pair: same encoder, same readout, same gate budget. The only
difference is the optimizer, one with a gradient and one without. Every gate can be a NAND, and
NANDs alone can build any circuit, so the genetic search could in principle find anything backprop
finds. What sets them apart is the search, not the model.

## Running the scorer

Scoring needs `yosys` (with ABC) and the sky130 library, which are not on pip:

```bash
conda create -n eda -c conda-forge -c litex-hub yosys open_pdks.sky130a
export MNISTBENCH_YOSYS=.../eda/bin/yosys
export MNISTBENCH_LIBERTY=.../sky130_fd_sc_hd__tt_025C_1v80.lib
python -m mnistbench.selftest     # emit, synthesize, simulate, bit-exact
```
