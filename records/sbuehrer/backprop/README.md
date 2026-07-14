# sbuehrer/backprop

**Gradient descent learns both what each gate *is* and how it is *wired*.**

A LUT gate needs two answers: which function am I, and which two signals do I read. Both are
discrete, and both are learned here — each as a hard choice in the forward pass with a smooth
gradient behind it.

**What (the truth table).** Four latent reals per gate, one per truth-table entry, binarized by
a straight-through estimator on a `sin`:

```python
hard = (sin(z) > 0)                      # exact 0/1  -> what the forward pass uses
soft = 0.5 + 0.5*sin(z)                  # smooth     -> what the gradient uses
bit  = hard + (soft - soft.detach())     # forward = hard, backward = d(soft)
```

`sin` rather than `sigmoid` because it is periodic: a latent never saturates, so there is always
a gradient pointing at the nearest 0/1 basin.

**Where (the connections).** Each of a gate's two inputs gets **8 candidate source signals**,
drawn at random once, plus a learnable logit per candidate. The forward pass selects the
**argmax** candidate — one wire, one exact bit — while the backward pass sees the **softmax**
over all 8, so a candidate that would have helped still receives gradient and the choice can
move during training.

```python
sel  = onehot(argmax(logits))            # one real wire (forward)
soft = softmax(logits)                    # smooth over the 8 candidates (backward)
wire = sel + (soft - soft.detach())
```

Selecting with a one-hot over *bits* keeps the forward pass exactly boolean. This matters
concretely: a softmax **mixture** of the 8 candidate bits would be a fraction, it has no
hardware, and the harness — which checks the python model against the synthesized netlist —
would reject the point. The accuracy printed during training is the accuracy of the silicon.

The head is a group popcount: the last layer's bits are cut into 10 groups, each group's ones
are counted, and the biggest count wins. In hardware that is an adder tree plus a comparator
chain, and it is in the gate count like everything else.

The encoder is a thermometer at thresholds `2^k - 1`. `pix > 127` is bit 7 of the byte, i.e. a
wire that costs zero gates — which is why the 1-bit points are so cheap on the x-axis.

[sbuehrer/genetic](../genetic) is the mirror image: it learns *only* the wiring, by mutation,
with no gradients at all.

## Points

`bits` thermometer bits per pixel, `widths` = gates per layer (the last is the readout, so it
must be divisible by 10). Training runs with a cosine schedule and early-stops when validation
has not improved for `patience` epochs, so every point is trained to convergence rather than to
a fixed epoch count.

```bash
python -m mnistbench run records/sbuehrer/backprop --device cuda
```
