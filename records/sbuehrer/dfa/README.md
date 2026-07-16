# sbuehrer/dfa

Fixed butterfly wiring, gate truth tables learned by **direct feedback alignment**. No
backpropagation, no learned connections.

## Architecture

A layered fan-in-2 LUT net whose wiring is fixed and never touched by the optimizer. The forward
wiring is a butterfly (FFT) pattern: gate `j` reads `j` and `j ^ (1 << k)`, with the stride `k`
halving every layer, so the receptive field doubles per layer. The stride must vary: a constant
stride pairs `j` with `j ^ (w/2)` forever, a 2-cycle that mixes nothing.

The encoder is a thermometer (`bits=1`, i.e. `pix > 127`, is bit 7 of the byte: a wire, zero gates).
The head is the shared group popcount + argmax. Only the 4 bits of each gate's truth table are
learned.

Every point is 5 layers deep and spends its silicon on the readout, which is measured rather than
assumed. Depth is a cost with no benefit here: a net whose gates each see 16 of 784 pixels beats one
where every gate sees all 784, because DFA's decay with depth dominates any receptive-field benefit.
The readout is the real lever. Widening it is worth ~+20 points for gates that are a rounding error
next to the body, because it is the one layer whose delta is the true gradient rather than a random
projection.

## Optimizer

A fixed random matrix `B_l` projects the output error *directly* onto layer `l`. No error signal
ever crosses a layer boundary, and no chain rule is ever applied between layers:

```
e       = softmax(votes/tau) - onehot(y)     # (B,10), the only global signal there is
delta_l = e @ B_l                            # (B,w), B_l fixed random (10,w)
G[i,p]  = sum_b delta_l[b,i] * 1[p_bi == p]  # scatter_add, (w,4)
z.grad  = G * 0.5*cos(z)                     # chain THROUGH a gate, never ACROSS one
```

DFA fits a LUT net unusually well. The forward pass is already exact bits, so a gate's output is
just `T[p]` where `p = 2a+b` is the pattern its inputs present. The derivative w.r.t. the gate's own
table is therefore the **indicator of the active pattern** (only `T[p]` moves), so each layer's
update is one `scatter_add` needing nothing but its own forward patterns and the broadcast error.
Layers can be updated in any order, or in parallel; they never talk.

`.backward()` is never called and no autograd graph is ever built: the whole step runs under
`torch.no_grad()`, so the no-backprop claim is structural rather than merely intended. Adam and the
cosine schedule are stock; only the gradient is hand-written. `lr=0.01` sits on a flat peak.

`B_l` is a training-time object. It is never synthesized, so it costs **zero gate equivalents**; the
structure that becomes silicon is the butterfly. The forward pass is exact boolean, so the val
accuracy printed during training *is* the circuit's accuracy.

## Points

`bits` thermometer bits per pixel, `width` = gates per body layer, `layers` = body layers, `readout`
= final layer width. Wiring is fixed; only the tables are learned. `epochs` is a ceiling;
validation early-stopping decides where each point stops.

| point | bits | width | layers | readout |
|---|---|---|---|---|
| xs | 1 | 256 | 5 | 640 |
| s | 1 | 512 | 5 | 1,280 |
| m | 1 | 1,024 | 5 | 5,120 |
| l | 1 | 2,048 | 5 | 10,240 |
| xl | 1 | 4,096 | 5 | 20,480 |

The knob is `layers`, **not** `depth`: `bench.merge_record` merges the measured fields over the
`POINTS` dict, and one of them is `depth` (the synthesized netlist's longest-path level count). A
`POINTS` key named `depth` is silently overwritten, and `results.json` then reports depth 192 for a
5-layer net, leaving the record unbuildable from its own results.

```bash
python -m mnistbench run records/sbuehrer/dfa --device cuda
```
