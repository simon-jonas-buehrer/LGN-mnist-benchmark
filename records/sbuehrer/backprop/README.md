# sbuehrer/backprop

**Learn what each gate *is*; leave the wiring alone.**

A difflogic-style LUT network. Each neuron reads two bits (fan-in 2) through a random,
permanently frozen wire pair, and applies a 2-input boolean function stored as a 4-entry truth
table. Four latent reals per gate, one per truth-table entry, are the only parameters. Adam
trains them.

The interesting part is that the forward pass is **already an exact boolean circuit** -- there
is no "train soft, then discretize and hope" step:

```python
hard = (sin(z) > 0)                      # exact 0/1  -> what the forward pass uses
soft = 0.5 + 0.5*sin(z)                  # smooth     -> what the gradient uses
bit  = hard + (soft - soft.detach())     # forward = hard, backward = d(soft)
```

`sin` rather than `sigmoid` because it is periodic, so a latent never saturates: there is
always a gradient pointing at the nearest 0/1 basin. The validation accuracy printed during
training is therefore the accuracy of the silicon, and the harness's python-vs-netlist check
passes exactly, not approximately.

The head is a group popcount: the last layer's bits are cut into 10 groups and each group's
ones are counted; the biggest count wins. In hardware that is an adder tree plus a comparator
chain, and it is in the gate count like everything else.

Encoder: a thermometer at thresholds `2^k - 1`. `pix > 127` is bit 7 of the byte, i.e. a wire
that costs nothing, which is why the 1-bit points are so cheap on the x-axis.

## Points

`bits` thermometer bits per pixel, `widths` = gates per layer (the last is the readout, so it
must be divisible by 10).

| point | bits | widths | epochs |
|---|---|---|---|
| xs | 1 | 320, 160 | 30 |
| s | 1 | 1280, 640 | 30 |
| m | 3 | 5120, 2560 | 40 |
| l | 3 | 16000, 8000, 4000 | 40 |
| xl | 7 | 48000, 24000, 12000 | 50 |

```bash
python -m mnistbench run records/sbuehrer/backprop --device cuda
```
