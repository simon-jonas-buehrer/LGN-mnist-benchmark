# sbuehrer/hebbian

Fixed butterfly wiring, LUT truth tables learned by a supervised Hebbian / three-factor rule. No
backpropagation, no backward pass of any kind, and — unlike `dfa` — no feedback pathway at all.

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

## What the sweeps said (val, never test)

This record shipped with the wrong shape. Its original points were 11–14 body layers deep with a
narrow readout, and every one of those choices measured backwards.

**Depth here is not a cost with no benefit — it is actively destructive.** At a matched gate budget:

| gates | shape | receptive field | val |
|---|---|---|---|
| 12,928 | w2048 d6 ro640 | 128/784 | **57.25%** |
| 33,088 | w4096 d8 ro320 | 432/784 | 50.61% |
| 49,312 | w4096 d12 ro160 | 784/784 | 38.23% |
| 28,832 | w2048 d14 ro160 | 784/784 | 20.78% |

Deep nets lose *despite* seeing the whole image, while shallow nets whose gates see 16 of 784 pixels
win. On full data the gap is stark: `w4096 d13 ro640` plateaus at **63.0%** after 40 epochs, where
`w4096 d3 ro10240` reaches **87.6%**.

The reason is structural, and it is the honest finding of this record. Every hidden layer is trained
against a *fixed random class assembly*, so a layer-1 gate is asked to predict the digit from two
pixels. That is not a hard problem, it is an impossible one: the gate fits noise, and every further
layer compounds the noise it is handed. The butterfly body destroys information rather than building
features. What classifies is the grouped vote at the end.

**The readout is the entire lever.** At a fixed ~9k gates, only the vote width moves the number:

| readout | 160 | 320 | 1280 |
|---|---|---|---|
| val (w2048 d4) | 43.65 | 58.47 | **71.80** |

Widening the vote is worth +28 points; the body is worth roughly nothing. So every point here is 3
layers deep and spends its silicon on the readout.

**`bits=1` wins**, and is free (`pix > 127` is bit 7, a wire). Thermometer bits only appear to help
when the readout is already starved — at `ro=5120`, `bits=3` (83.58%) and `bits=1` (81.77%) land in
the same place for more area, so the extra encoder bits buy nothing.

These are the same three lessons the `dfa` record reached independently on this same butterfly, which
is some evidence they are properties of the wiring family rather than of either rule.

## Points

`bits` thermometer bits per pixel, `width` = gates per body layer, `layers` = body layers, `readout`
= final grouped vote layer. Wiring is fixed; only the tables are learned.

| point | bits | width | layers | readout | GE | test acc |
|---|---|---|---|---|---|---|
| xs | 1 | 512 | 3 | 640 | 5,752 | 67.21% |
| s | 1 | 1,024 | 3 | 2,560 | 18,322 | 80.04% |
| m | 1 | 2,048 | 3 | 5,120 | 33,958 | 84.62% |
| l | 1 | 4,096 | 3 | 10,240 | 71,640 | 88.48% |
| xl | 1 | 8,192 | 3 | 20,480 | 135,783 | 90.61% |

The knob is `layers`, **not** `depth`: `bench.run_point` merges the measured netlist fields over the
`POINTS` dict, and one of them is `depth` (the synthesized netlist's longest-path level count). A
`POINTS` key named `depth` is silently overwritten, and `results.json` then reports the netlist depth
in place of the body depth — leaving the record unbuildable from its own results. This record shipped
with that bug and inherited the fix from `dfa`.

```bash
python -m mnistbench run records/sbuehrer/hebbian --device cuda
```

## Where it lands, honestly

Hebbian is **below the frontier at every size**, and every one of its five points is dominated by
`forest` — at `l`, 71,640 GE buys 88.48% where a boosted forest gets 97.20% for *two thirds* of the
area. Against a real optimizer this rule is not competitive, and no amount of tuning inside this
design closes an 8-point gap.

The comparison that is worth drawing is against `dfa`, which holds the wiring fixed in exactly the
same way and differs only in what teaches the tables. Interpolating `dfa` to the same area:

| | xs | s | m | l | xl |
|---|---|---|---|---|---|
| GE | 5,752 | 18,322 | 33,958 | 71,640 | 135,783 |
| hebbian | **67.21** | 80.04 | 84.62 | 88.48 | 90.61 |
| dfa (interp.) | 66.30 | 81.95 | 86.76 | 90.66 | 92.67 |
| delta | **+0.91** | −1.91 | −2.14 | −2.18 | −2.06 |

Deleting the feedback pathway altogether costs about **two points of accuracy at matched area** —
and at the smallest size costs nothing at all. That is the result: on this butterfly, a fixed random
projection of the output error (`dfa`) and a fixed random class assembly per layer (here) are
teachers of nearly the same quality, because neither one carries information *about the layer it is
teaching*. What separates both from `backprop` is much larger than what separates them from each
other.

The ceiling is the hidden-layer target, not the tuning. Every hidden gate is asked to predict the
class from two bits, so the body cannot build features and the readout does the work — which is why
these points buy accuracy almost entirely with vote width. A rule that gave hidden layers a target
that is *not* the class (a decorrelating or unsupervised objective) is the change worth making, and
it is a different record rather than another point on this curve.

The forward pass is exact boolean, so the val accuracy printed during training *is* the circuit's
accuracy, and the harness's 512-image model-vs-netlist check passes for every point.
