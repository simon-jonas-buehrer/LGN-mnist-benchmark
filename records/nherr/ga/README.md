# nherr/ga

Learn the truth tables *and* the wiring, by a population. No gradients.

Every gate is a 2-input LUT with its own 4-bit truth table, and each of its two ports picks one of
`K = 8` seeded candidate sources (the **codebook**: 3 bits per wire instead of `log2(fan-in)`).
Table bits and wire choices sit in one genome, and a generational GA searches both at once.

```
each generation:
    tournament-select pop parents on one fresh minibatch
    gate-wise crossover: for each gate, take (table, wire_a, wire_b) from one parent or the other
    mutate: flip table bits, rewire ports; both rates annealed
    keep the top `elite` unchanged
```

What the record found, on its own axis, before it was measured here:

* **Crossover is the whole advantage** (+6.5 points). Every variant tried on top of plain gate-wise
  crossover — importance-biased, cone, GOMEA — came back inside noise or worse, and the ordering
  across budget-matched algorithms was `ga` > `snes` > `eda` > `ga_nox` > `aging` > `gomea` >
  `mapelites` > `nslc`.
* **The codebook is nearly free.** It gets ~97% of free wiring's gain at 3 bits per wire.
* **65% of the gates end up dead**, and that is the search's choice rather than an architecture
  cap: they neither help (the scratch-space hypothesis was tested and rejected) nor hurt. On this
  axis the finding is subsumed rather than rewarded — ABC deletes dead logic before charging area.
* **Width does not pay.** At equal evaluations, wider is worse; the search is data-limited, not
  capacity-limited. That reproduces here: `l` is dominated by `m`.

The comparison this record exists to make is against `sbuehrer/genetic`, which fixes every gate to
NAND and evolves only the wiring: learned tables + codebook wiring + crossover wins at every size,
by 5-8 points. Against `sbuehrer/backprop` it wins below ~10k GE and loses above. Against
`sbuehrer/forest` it loses above ~5k GE — a logic net is not automatically the cheap way to do
MNIST.

## Points

`bits` thermometer bits per pixel, `widths` = gates per layer (the last is the readout, so it must
be divisible by 10). All four use the same budget: pop 512, 20k generations, fitness =
margin + 100 x batch accuracy.

| point | bits | widths | GE | test acc |
|---|---|---|---|---|
| xs | 1 | 512, 256, 160 | 2,109 | 88.21% |
| s | 3 | 1536, 512, 320 | 4,214 | 90.17% |
| m | 7 | 3072, 1024, 500 | 7,471 | 91.37% |
| l | 7 | 6144, 2048, 1000 | 14,853 | 91.26% |

Thresholds are `hw.even_thresholds`, which land on 2^k-1 boundaries: `pix > 127` is bit 7 of the
byte, a wire, and costs nothing. The record's own default (32, 64, 96, ...) is one grey level off
that and would make all seven thresholds real comparators across 784 pixels, for no accuracy.

`m` is the headline configuration; it reads 91.7% on the record's own split and 91.37% here, under
a different protocol. Do not quote the two interchangeably.

## The code

Three files, and nothing that is not on the path from MNIST to a measured circuit.

| file | what it is |
|---|---|
| `lutnet.py` | the circuit: packed forward, codebook wiring, popcount readout |
| `ga.py` | the search: `Config`, mutation, crossover, `train` |
| `submission.py` | the harness contract: encode, train, `predict`, `emit_verilog` |

The upstream record also carried seven rival algorithms, three gate-importance measures, a
dead-gate autopsy, a bit-exact pruner and an STE-backprop counterpart. None of them is on this
path, so none of them is here; the findings above are what they established. Both files are carved
verbatim out of those originals — the GA this ships trains bit-for-bit the same circuit the four
measured points were built from.

```bash
pip install -e '.[jax]'
python -m mnistbench run records/nherr/ga --device cuda
```
