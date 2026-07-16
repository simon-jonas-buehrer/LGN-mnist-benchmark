# sbuehrer/bitnet

A ternary-weight, binary-activation MLP, translated fully into gates.

Weights are ternary (-1, 0, +1) as in BitNet; activations are single bits. A neuron is a
ternary-weighted count of its input bits, thresholded:

```
h_j = [ (# inputs with weight +1 that are 1) - (# with weight -1) + b_j  >  0 ]
```

In silicon that is two popcounts, a subtract and a comparator, i.e. an adder tree, not a lookup
table. Putting it on this benchmark shows what dense ternary arithmetic costs in gate-equivalents
next to a learned gate net.

Training is straight-through: the latent real weights are ternarized (TWN threshold at
`0.7 * mean|w|`) and the bias is rounded to an integer on the forward pass, with the gradient
passed straight through. The forward pass is therefore integer-exact, so `predict()` equals the
synthesized netlist bit for bit. The encoder is a thermometer and the head is the shared group
popcount + argmax, so only the hidden layers are ternary.

## Points

`bits` thermometer bits per pixel, `hidden` = ternary hidden-layer widths, `readout` = final
ternary layer width (divisible by 10). Training early-stops on validation.

Each point is the best width/depth for its gate budget, from a validation sweep.

| point | bits | hidden | readout |
|---|---|---|---|
| xs | 1 | 64 | 320 |
| s | 1 | 128 | 320 |
| m | 1 | 256, 256 | 320 |
| l | 1 | 512 | 640 |
| xl | 3 | 512, 512 | 640 |

Ternary arithmetic is dense, so these land in the millions of gate equivalents, far above the
logic-gate records. That is the point of the comparison.

```bash
python -m mnistbench run records/sbuehrer/bitnet --device cuda
```
