# sbuehrer/backprop

Gradient descent learns both what each gate computes and how it is wired.

A LUT gate needs two answers: which function it is, and which two signals it reads. Both are
discrete, and both are learned as a hard choice in the forward pass with a smooth gradient behind
it.

The truth table. Four latent reals per gate, one per truth-table entry, binarized by a
straight-through estimator on a `sin`:

```python
hard = (sin(z) > 0)                      # exact 0/1, what the forward pass uses
soft = 0.5 + 0.5*sin(z)                  # smooth, differentiable
bit  = hard + (soft - soft.detach())     # forward = hard, backward = d(soft)
```

`sin` rather than `sigmoid` because it is periodic: a latent never saturates, so there is always a
gradient toward the nearest 0/1 basin.

The connections. Each of a gate's two inputs gets 8 candidate source signals, drawn at random once,
plus a learnable logit per candidate. The forward pass takes the argmax candidate (one real wire,
one exact bit); the backward pass sees the softmax over all 8, so a candidate that would have
helped still gets gradient and the choice can move.

Selecting a one-hot over bits keeps the forward pass exactly boolean. A softmax mixture of the 8
candidate bits would be a fraction, which has no hardware, and the harness would reject the point.
The accuracy printed during training is the accuracy of the silicon.

The head is a group popcount: the last layer's bits are cut into 10 groups, each group's ones are
counted, and the largest count wins. In hardware that is an adder tree and a comparator chain, and
it is counted like everything else. The encoder is a thermometer at thresholds `2^k - 1`, so
`pix > 127` is bit 7 of the byte, a wire that costs no gates.

`lr = 0.2`, `batch = 128`, from a sweep on the `m` point. The peak is flat: any learning rate from
0.02 to 0.2 lands within about 0.3 points, so the result does not depend on a lucky setting.

## Points

`bits` thermometer bits per pixel, `widths` = gates per layer (the last is the readout, so it must
be divisible by 10). `epochs` is a ceiling; training early-stops at its own convergence.

| point | bits | widths |
|---|---|---|
| xs | 1 | 320, 160 |
| s | 1 | 1280, 640 |
| m | 3 | 5120, 2560 |
| l | 3 | 16000, 8000, 4000 |
| xl | 7 | 48000, 24000, 12000 |

```bash
python -m mnistbench run records/sbuehrer/backprop --device cuda
```
