| | record | point | gate equivalents | depth | MNIST test acc | test CE |
|---|---|---|---|---|---|---|
| * | `sbuehrer/backprop` | l | 52,973 | 238 | **95.35%** | 0.152 |
| * | `sbuehrer/backprop` | m | 32,425 | 237 | **93.41%** | 0.213 |
| * | `sbuehrer/backprop` | s | 7,514 | 188 | **87.89%** | 0.390 |
| * | `sbuehrer/genetic` | xs | 1,945 | 129 | **80.38%** | 0.627 |
| * | `sbuehrer/backprop` | xs | 1,913 | 130 | **73.10%** | 0.782 |

`*` = on the Pareto frontier (nothing is both smaller and more accurate). test CE = temperature-calibrated cross-entropy over the circuit's class votes.
