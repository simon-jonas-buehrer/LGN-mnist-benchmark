# sbuehrer/genetic

Learn how the gates are wired; every gate is a NAND.

Each gate is fixed to NAND, which is functionally complete, so this search space contains every
circuit the LUT net can express. The only free parameters are which two signals each gate reads.
No gradients.

```
each generation:
    make k-1 mutants of the current wiring (rewire mut gate endpoints at random)
    score all k, the incumbent included, on the same minibatch
    keep the best
```

Three details that matter:

* Fitness is a margin, not accuracy. Minibatch accuracy changes only when a prediction flips, so
  almost every single-wire mutation scores the same and the search random-walks. The margin (votes
  for the true class minus the best wrong class) moves whenever any vote moves, which turns the
  plateau into a slope.
* The selection batch must be big. One rewired wire moves the margin by a hair; a small batch
  buries that in sampling noise, so selection keeps the luckier mutant rather than the better one.
  It is worth tens of points, not a few: on `xs`, batch 1024 lands at 22.6% where batch 8192 lands
  at 60.4%, nothing else changed. `batch=16384` is where it saturates.
* Delta forward. A mutant differs from the incumbent only from its lowest mutated layer upward, so
  every layer below is reused. Exact, and it is most of the speed.

The encoder and readout head are identical to the backprop record, so the two curves differ only
in what is learned. Each point trains until validation stops improving; `gens` is a ceiling.

## Points

`bits` thermometer bits per pixel, `widths` = gates per layer (the last is the readout, so it must
be divisible by 10).

| point | bits | widths |
|---|---|---|
| xs | 1 | 256, 256, 160 |
| s | 1 | 1024, 1024, 320 |
| m | 3 | 2048, 2048, 2048, 640 |
| l | 3 | 4096 x 4, 1280 |

`l` is the top of this curve, and it is a wall rather than a budget. A fifth point at
`8000, 8000, 8000, 2400` ran 16 GPU-hours to 525k generations, reached 86.2% val and was still
gaining ~0.02 at a time, while `l` gets 87.3% for a third of the area: bigger, slower, and worse.
The wider the net, the less one rewired wire moves the margin, which is the same effect the curve
flattens under. So the record stops at `l`.

```bash
python -m mnistbench run records/sbuehrer/genetic --device cuda
```
