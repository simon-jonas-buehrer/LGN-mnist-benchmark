| | record | point | gate equivalents | depth | MNIST test acc | test CE |
|---|---|---|---|---|---|---|
| * | `sbuehrer/forest` | xxl | 61,304 | 229 | **97.31%** | 0.081 |
|  | `sbuehrer/backprop` | xl | 156,861 | 285 | **96.93%** | 0.102 |
| * | `sbuehrer/forest` | xl | 29,776 | 235 | **96.82%** | 0.102 |
| * | `sbuehrer/forest` | l | 15,449 | 206 | **96.08%** | 0.121 |
|  | `sbuehrer/backprop` | l | 52,973 | 238 | **95.35%** | 0.152 |
|  | `sbuehrer/bitnet` | l | 2,111,700 | 601 | **94.71%** | 0.208 |
| * | `sbuehrer/forest` | m | 7,711 | 188 | **93.99%** | 0.201 |
|  | `sbuehrer/bitnet` | m | 1,156,908 | 682 | **93.71%** | 0.257 |
|  | `sbuehrer/backprop` | m | 32,425 | 237 | **93.41%** | 0.213 |
|  | `sbuehrer/dfa` | xl | 174,903 | 313 | **93.40%** | 0.216 |
|  | `sbuehrer/bitnet` | s | 482,469 | 457 | **91.96%** | 0.328 |
|  | `sbuehrer/dfa` | l | 91,506 | 282 | **91.53%** | 0.265 |
|  | `sbuehrer/hebbian` | xl | 135,783 | 284 | **90.61%** | 0.304 |
|  | `sbuehrer/bitnet` | xs | 233,380 | 439 | **90.57%** | 0.361 |
|  | `sbuehrer/dfa` | m | 45,455 | 244 | **89.04%** | 0.331 |
| * | `sbuehrer/forest` | s | 3,027 | 141 | **88.59%** | 0.389 |
|  | `sbuehrer/hebbian` | l | 71,640 | 270 | **88.48%** | 0.365 |
|  | `sbuehrer/backprop` | s | 7,514 | 188 | **87.89%** | 0.390 |
|  | `sbuehrer/genetic` | l | 20,114 | 206 | **87.28%** | 0.422 |
|  | `sbuehrer/genetic` | m | 9,146 | 191 | **86.52%** | 0.444 |
|  | `sbuehrer/hebbian` | m | 33,958 | 239 | **84.62%** | 0.470 |
|  | `sbuehrer/genetic` | s | 3,920 | 156 | **83.16%** | 0.558 |
| * | `sbuehrer/forest` | xs | 1,577 | 132 | **81.92%** | 0.593 |
|  | `sbuehrer/genetic` | xs | 1,945 | 129 | **80.38%** | 0.627 |
|  | `sbuehrer/hebbian` | s | 18,322 | 227 | **80.04%** | 0.604 |
|  | `sbuehrer/dfa` | s | 13,036 | 206 | **79.29%** | 0.612 |
|  | `sbuehrer/backprop` | xs | 1,913 | 130 | **73.10%** | 0.782 |
|  | `sbuehrer/hebbian` | xs | 5,752 | 177 | **67.21%** | 0.970 |
| * | `sbuehrer/forest` | xxs | 676 | 115 | **67.07%** | 1.062 |
|  | `sbuehrer/dfa` | xs | 6,776 | 192 | **66.30%** | 0.978 |

`*` = on the Pareto frontier (nothing anywhere is both smaller and more accurate). Each record is listed as its own frontier: a point a record already beats on both axes is dropped, so a record shows one row per size it is best at. test CE = calibrated cross-entropy over the circuit's class votes.
