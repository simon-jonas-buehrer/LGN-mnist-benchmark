# sbuehrer/hebbian

Fixed butterfly wiring, LUT truth tables learned by a supervised Hebbian / three-factor rule.

Each gate sees only two input bits, so a trainable gate has four possible local patterns.  During
training, the active pattern is the eligibility trace.  The label supplies a clamped target
assembly, while the network's current class probabilities supply the free phase:

```text
third factor = target_class_code - predicted_class_mixture
table[p]    += eligibility[p] * third factor
```

Hidden layers get fixed sparse class assemblies.  The readout layer gets the usual one-vs-rest
class groups.  A hidden gate therefore learns whether its local two-bit pattern should participate
in the label's assembly, and the readout learns which group should fire.

No autograd graph is built and no `.backward()` call is made.  The update is just scatter-counting
onto the active truth-table entry.  The forward pass is hard boolean throughout, so the validation
accuracy printed during training is the exact circuit accuracy before synthesis.

## Points

`bits` thermometer bits per pixel.  `width` is the fixed butterfly body width, `depth` is the
number of body layers, and `readout` is the final grouped vote layer.

| point | bits | width | depth | readout |
|---|---:|---:|---:|---:|
| xs | 1 | 1024 | 11 | 320 |
| s | 1 | 2048 | 12 | 640 |
| m | 3 | 4096 | 13 | 640 |
| l | 3 | 8192 | 14 | 1280 |

```bash
python -m mnistbench run records/sbuehrer/hebbian --device cuda
```
