| | record | point | gate equivalents | depth | MNIST test acc | test CE |
|---|---|---|---|---|---|---|
| * | `sbuehrer/forest` | xl | 47,794 | 257 | **97.20%** | 0.086 |
|  | `sbuehrer/backprop` | xl | 156,861 | 285 | **96.93%** | 0.102 |
| * | `sbuehrer/forest` | l | 20,272 | 235 | **96.18%** | 0.119 |
|  | `sbuehrer/backprop` | l | 52,973 | 238 | **95.35%** | 0.152 |
| * | `sbuehrer/forest` | m | 7,711 | 188 | **93.99%** | 0.201 |
|  | `sbuehrer/backprop` | m | 32,425 | 237 | **93.41%** | 0.213 |
|  | `sbuehrer/dfa` | l | 91,506 | 282 | **91.53%** | 0.265 |
|  | `sbuehrer/dfa` | m | 45,455 | 244 | **89.04%** | 0.331 |
| * | `sbuehrer/forest` | s | 3,027 | 141 | **88.59%** | 0.389 |
|  | `sbuehrer/backprop` | s | 7,514 | 188 | **87.89%** | 0.390 |
|  | `sbuehrer/genetic` | l | 20,114 | 206 | **87.28%** | 0.422 |
|  | `sbuehrer/genetic` | m | 9,146 | 191 | **86.52%** | 0.444 |
| * | `sbuehrer/forest` | xs | 1,848 | 142 | **84.69%** | 0.538 |
|  | `sbuehrer/genetic` | s | 3,920 | 156 | **83.16%** | 0.558 |
|  | `sbuehrer/genetic` | xs | 1,945 | 129 | **80.38%** | 0.627 |
|  | `sbuehrer/dfa` | s | 13,036 | 206 | **79.29%** | 0.612 |
|  | `sbuehrer/backprop` | xs | 1,913 | 130 | **73.10%** | 0.782 |
| * | `sbuehrer/forest` | tiny | 676 | 115 | **67.07%** | 1.062 |
|  | `sbuehrer/dfa` | xs | 6,776 | 192 | **66.30%** | 0.978 |

`*` = on the Pareto frontier (nothing is both smaller and more accurate). test CE = calibrated cross-entropy over the circuit's class votes.
