# sbuehrer/targetprop

Fixed Monarch wiring, gate truth tables learned without gradients.

The other two records learn the wiring. This one fixes it in a Monarch pattern (block-diagonal
within groups on even layers, across groups on odd layers), which reaches full receptive field in
log-depth, and learns only the 2-input truth table of each gate. No gradients.

Each iteration, on a large batch:

1. Run the discrete net, caching every gate's input pattern.
2. At the readout, ask each class group for a few bit flips that would widen the margin between the
   true class and the best wrong class. Margin, not accuracy, so the signal is a slope.
3. Propagate those targets down. A gate with a target either already outputs it, or it does not, in
   which case we both vote that its truth-table entry should change, and push a target upstream to
   the input that, if flipped, would fix it. Change the gate or change its inputs, decided by the
   votes.
4. After the batch, update each table entry by majority vote, through a small real-valued
   accumulator so a bit only flips once a consistent vote builds up. The emitted table stays exactly
   boolean, so the circuit matches `predict()` bit for bit.

Residual init (every gate passes input A) makes the net start as identity. The accumulator keeps
the latents unsaturated so votes can move them; a plain majority-flip would sit stuck at identity.

## Points

`bits` thermometer bits per pixel, `width` = gates per body layer, `depth` = body layers, `readout`
= final layer width (divisible by 10). Wiring is fixed; only the tables are learned. Training
early-stops when validation stops improving.

| point | bits | width | depth | readout |
|---|---|---|---|---|
| xs | 1 | 1024 | 20 | 320 |
| s | 1 | 2048 | 22 | 640 |
| m | 3 | 4096 | 24 | 640 |
| l | 3 | 8192 | 26 | 1280 |

```bash
python -m mnistbench run records/sbuehrer/targetprop --device cuda
```
