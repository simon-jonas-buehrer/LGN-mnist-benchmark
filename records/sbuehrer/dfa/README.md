# sbuehrer/dfa

Fixed butterfly wiring, gate truth tables learned by **direct feedback alignment**. No
backpropagation, no learned connections.

Both of the network's structures are fixed and non-learnable. The forward wiring is a butterfly
(FFT) pattern: gate `j` reads `j` and `j ^ (1 << k)`, with the stride `k` halving every layer. The
backward pathway is a fixed random matrix `B_l` that projects the output error *directly* onto layer
`l`. Only the 4 bits of each gate's truth table move.

No error signal ever crosses a layer boundary, and no chain rule is ever applied between layers:

```
e       = softmax(votes/tau) - onehot(y)     # (B,10), the only global signal there is
delta_l = e @ B_l                            # (B,w), B_l fixed random (10,w)
G[i,p]  = sum_b delta_l[b,i] * 1[p_bi == p]  # scatter_add, (w,4)
z.grad  = G * 0.5*cos(z)                     # chain THROUGH a gate, never ACROSS one
```

`.backward()` is never called and no autograd graph is ever built — the whole step runs under
`torch.no_grad()`, so the no-backprop claim is structural rather than merely intended. Adam and the
cosine schedule are stock; only the gradient is hand-written.

DFA fits a LUT net unusually well. The forward pass is already exact bits, so a gate's output is
just `T[p]` where `p = 2a+b` is the pattern its inputs present. The derivative w.r.t. the gate's own
table is therefore the **indicator of the active pattern** — only `T[p]` moves — so each layer's
update is one `scatter_add` needing nothing but its own forward patterns and the broadcast error.
Layers can be updated in any order, or in parallel; they never talk.

`B_l` is a training-time object. It is never synthesized, so it costs **zero gate equivalents**; the
structure that becomes silicon is the butterfly.

## What the sweeps said (val, never test)

Two measurements overturned the design this record started with.

**Depth is a cost with no benefit, and full mixing is counterproductive.** The plan was
`depth = log2(width)+1`, so every readout gate sees every pixel. Measured, that is exactly wrong:

| gates | shape | receptive field | val |
|---|---|---|---|
| 6,464 | w2048 d3 | 16/784 | **67.03%** |
| 11,584 | w1024 d11 | 784/784 | 60.47% |
| 14,656 | w1024 d14 | 784/784 | 57.40% |

A net whose gates each see 16 pixels beats one where every gate sees all 784. DFA's decay with depth
dominates any receptive-field benefit, so every point here is 5 layers deep.

**The readout is the real lever**, and it was starving the model. At w8192 d5:

| readout | 320 | 640 | 1280 | 2560 | 5120 | 10240 | 20480 |
|---|---|---|---|---|---|---|---|
| val | 72.67 | 78.45 | 83.52 | 87.27 | 89.45 | 91.20 | **92.95** |

+20 points for gates that are a rounding error next to the body. So the large points grow the
readout, not the body — the readout is the one layer whose delta is the true gradient rather than a
random projection, and it is where a DFA net wants its silicon.

`bits=1` wins on **both** axes (most accurate *and* free: `pix > 127` is bit 7, a wire).
`lr=0.01` is a flat peak. Width saturates around 8192.

## Points

`bits` thermometer bits per pixel, `width` = gates per body layer, `layers` = body layers, `readout`
= final layer width. Wiring is fixed; only the tables are learned.

| point | bits | width | layers | readout | GE | test acc |
|---|---|---|---|---|---|---|
| xs | 1 | 256 | 5 | 640 | 6,776 | 66.30% |
| s | 1 | 512 | 5 | 1,280 | 13,036 | 79.29% |
| m | 1 | 1024 | 5 | 5,120 | 45,455 | 89.04% |
| l | 1 | 2048 | 5 | 10,240 | 91,506 | 91.53% |

```bash
python -m mnistbench run records/sbuehrer/dfa --device cuda
```

## Where it lands, honestly

DFA is **below** the frontier at every size. At ~13k GE it gets 79.3% where `backprop` gets 87.9%
for *half* the area; `forest` reaches 96.2% at 20k GE, which `dfa` does not approach at 4x the area.

That gap is the result, not a failure to tune. This record removes two things at once — the learned
wiring *and* the backward pass — and the curve prices what they were worth. The comparison worth
drawing is against the `targetprop` record (removed in 03581e7), which held the wiring fixed exactly
this way and tried to learn the tables from propagated targets instead: it never left chance, on this
same butterfly, because a constant truth table is an absorbing state it could not climb out of. A
random projection of the output error is a weak teacher, but it is a teacher.

The forward pass is exact boolean, so the val accuracy printed during training *is* the circuit's
accuracy, and the harness's 512-image model-vs-netlist check passes for every point.
