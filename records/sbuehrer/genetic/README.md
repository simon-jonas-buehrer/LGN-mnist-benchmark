# sbuehrer/genetic

**Learn how the gates are *wired*; every gate is a NAND, forever.**

The mirror image of [sbuehrer/backprop](../backprop). There, each gate could become any of the
16 boolean functions but its two input wires were frozen. Here the function is fixed — NAND, which
is functionally complete, so this search space contains every circuit the LUT net can express —
and the only free parameters are which two signals each gate reads.

No gradients anywhere. The algorithm is the simplest thing that deserves the name:

```
each generation:
    make k-1 mutants of the current wiring (rewire `mut` gate endpoints at random)
    score all k -- the incumbent included -- on the SAME minibatch
    keep the best
```

Three details that are load-bearing, not cosmetic:

* **Fitness is a margin, not accuracy.** Minibatch accuracy only changes when a prediction
  flips, so almost every single-wire mutation scores identically and the search random-walks on
  a plateau. The margin (votes for the true class minus the best wrong class) moves whenever any
  vote moves, which turns the plateau into a slope.
* **The selection batch must be big.** One rewired wire moves the margin by a hair; a small
  batch buries that hair in sampling noise, and selection then keeps the *luckier* mutant rather
  than the *better* one. Measured on `xs`, 20k generations, nothing else changed:

  | selection batch | val accuracy |
  |---|---|
  | 1024 | 22.6% |
  | 4096 | 24.9% |
  | **8192** | **60.4%** |

  A 2.7x larger batch is worth 38 points. Nothing else in this search comes close, and it is the
  first thing to check before concluding that a gradient-free method "just doesn't work".
* **Delta forward.** A mutant differs from the incumbent only from its lowest mutated layer
  upward, so every layer below that is reused. Exact, and it is most of the speed.

Encoder and readout head are identical to the backprop record, so the two curves differ only in
what is being learned.

This record is here to be beaten. It is also here to show what a gradient-free search costs in
silicon: a hill-climber has to spend gates to make up for the gates it cannot aim.

## Points

`bits` thermometer bits per pixel, `widths` = gates per layer (the last is the readout, so it
must be divisible by 10).

| point | bits | widths | generations |
|---|---|---|---|
| xs | 1 | 256, 256, 160 | 20,000 |
| s | 1 | 1024, 1024, 320 | 30,000 |
| m | 3 | 2048, 2048, 2048, 640 | 40,000 |
| l | 3 | 4096 x 4, 1280 | 40,000 |

```bash
python -m mnistbench run records/sbuehrer/genetic --device cuda
```
