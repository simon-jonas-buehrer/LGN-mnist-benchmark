# sbuehrer/hebbian

Fixed butterfly wiring, LUT truth tables learned by a supervised Hebbian / three-factor rule. No
backpropagation, no backward pass of any kind, and, unlike `dfa`, no feedback pathway at all.

## Architecture

The same fixed fan-in-2 butterfly body as `dfa`: gate `j` reads `j` and `j ^ (1 << k)`, stride `k`
halving every layer. The encoder is a thermometer (`bits=1` is bit 7 of the byte, a wire) and the
head is the grouped class vote. Only the truth tables are learned.

Every point is 3 layers deep and spends its silicon on the readout. Depth here does not just fail to
pay for itself, it destroys accuracy: deep nets lose *despite* seeing the whole image, while shallow
nets whose gates see 16 of 784 pixels win. The reason is structural, and it is the honest
finding of this record. Every hidden layer is trained against a *fixed random class assembly*, so a
layer-1 gate is asked to predict the digit from two pixels. That is not a hard problem, it is an
impossible one: the gate fits noise, and every further layer compounds the noise it is handed. The
butterfly body destroys information rather than building features. What classifies is the grouped
vote at the end. Widening it is worth tens of points where the body is worth roughly nothing.

## Optimizer

Each gate sees only two input bits, so a trainable gate has four possible local patterns. During
training, the active pattern is the eligibility trace. The label supplies a clamped target assembly,
while the network's current class probabilities supply the free phase:

```text
third factor = target_class_code - predicted_class_mixture
table[p]    += eligibility[p] * third factor
```

Hidden layers get fixed sparse class assemblies. The readout layer gets the usual one-vs-rest class
groups. A hidden gate therefore learns whether its local two-bit pattern should participate in the
label's assembly, and the readout learns which group should fire.

No autograd graph is built and no `.backward()` call is made. The update is just scatter-counting
onto the active truth-table entry. The forward pass is hard boolean throughout, so the validation
accuracy printed during training is the exact circuit accuracy before synthesis.

## Points

`bits` thermometer bits per pixel, `width` = gates per body layer, `layers` = body layers, `readout`
= final grouped vote layer. Wiring is fixed; only the tables are learned. `epochs` is a ceiling;
validation early-stopping decides where each point stops.

| point | bits | width | layers | readout |
|---|---|---|---|---|
| xs | 1 | 512 | 3 | 640 |
| s | 1 | 1,024 | 3 | 2,560 |
| m | 1 | 2,048 | 3 | 5,120 |
| l | 1 | 4,096 | 3 | 10,240 |
| xl | 1 | 8,192 | 3 | 20,480 |

The knob is `layers`, **not** `depth`: a `POINTS` key named `depth` collides with the netlist depth
the harness measures and is silently overwritten. This record shipped with that bug and inherited
the fix from `dfa`.

```bash
python -m mnistbench run records/sbuehrer/hebbian --device cuda
```
