# dmuglich/did

Same circuit as `nherr/ga`, no population: one network, improved by discrete local search.

DID — discrete influence descent — replaces the population with a ranked proposal pool. Each sweep
draws a fresh batch and linearises the loss into a signed per-gate sensitivity, which turns into
candidate moves of three kinds:

* **table rows** — flip one row of one gate's truth table;
* **parent-child motifs** — for an edge `j -> k`, roll each of the 16 parent tables through cached
  activations and pair it with the child's closed-form best response, as one two-gate move;
* **codebook rewires** — point a port at a different one of the `K = 8` candidate sources, scored
  jointly with the best-response table for that gate; `did_joint` also scores all `K^2` pairs, the
  two-port moves the per-port bins cannot see.

All three rank in **one** global pool by surrogate delta. The top of that pool is then tried one at
a time by an exact forward pass, and a move is accepted only when the measured loss actually drops
— the linearisation proposes, it never decides. `did_dedup` keeps only the best-ranked proposal
per gate, because ~85% of a raw top-512 targets a gate some better-ranked entry already claimed and
would spend its trial on a stale genome.

Every trial is charged against the same evaluation budget the GA gets, so `ga` and `did` are
budget-matched rather than merely similar.

## Points

`bits` thermometer bits per pixel, `widths` = gates per layer (the last is the readout, so it must
be divisible by 10). Every point runs the converged configuration — rewire + joint + dedup, 100k
sweeps — and differs only in the net searched over. The shapes are the byte classes the record swept
upstream; `s` is its headline funnel, the rest are uniform.

| point | bits | widths | gates |
|---|---|---|---|
| xs | 7 | 620, 620, 600 | 1,840 |
| s | 7 | 3072, 1024, 500 | 4,596 |
| m | 7 | 2460, 2450, 2450 | 7,360 |
| l | 7 | 4400, 4400, 4400 | 13,200 |
| xl | 7 | 8800, 8800, 8800 | 26,400 |

Thresholds are `hw.even_thresholds`, as in `nherr/ga`: they land on 2^k-1 boundaries, so `pix > 127`
is bit 7 of the byte and costs no gates.

## Not measured yet

**This record has no `results.json`, so it is not on the board.** The upstream work measured DID on
a *bytes* axis — the bits a deployment carries — which prices the genome and nothing else; the
thermometer comparators and the popcount adder are free in bytes and real in silicon. Nobody has
run these five points through `mnistbench` yet.

For scale, and **not** as a benchmark number, here is what those same five shapes reached on that
other axis (self-reported accuracy on the full MNIST test set, label-trained, 100k sweeps — a
different split and a different protocol, so it is not comparable to a column in the leaderboard):

| point | bytes | accuracy on that axis |
|---|---|---|
| xs | 2,300 | 94.81% |
| s | 5,745 | 94.87% |
| m | 9,200 | 96.75% |
| l | 16,500 | 97.11% |
| xl | 33,000 | 97.27% |

If that survives synthesis and the fixed split, DID lands well above `nherr/ga` — which is the
comparison the record exists to make, since the two share the encoder, the readout, the genome and
the budget, and differ only in population-versus-local-search. Treat it as a hypothesis until a
point is measured.

The converged budget is expensive: 100k sweeps is 51.2M evaluations, about 4 GH200-hours at the `s`
shape and more above it. `gens=20000` is the budget-matched setting the GA comparison uses, and is
the cheaper thing to run first — edit `POINTS`, or measure one point at a time with `--point`.

## The code

Three files, and nothing that is not on the path from MNIST to a measured circuit.

| file | what it is |
|---|---|
| `lutnet.py` | the circuit: packed forward, codebook wiring, popcount readout |
| `did.py` | the search: `Config`, the influence/proposal/acceptance machinery, `Runner` |
| `submission.py` | the harness contract: encode, train, `predict`, `emit_verilog` |

`did.py` is carved verbatim out of a file that held eight budget-matched algorithms behind one
`Runner` — a GA and its variants, aging evolution, MAP-Elites, NSLC, an EDA, SNES, GOMEA, and a
hill-climber control — plus a teacher-distillation path and a DID unit-test suite. Only DID is
here. The comparison against a population now lives across the two records rather than inside one
file: `nherr/ga` is that GA.

`lutnet.py` is this record's own copy. `nherr/ga` carries a later fork of the same circuit, and
the two are not interchangeable.

`submission.py` uses the fixed 54k/6k/10k split and never touches `data.test_*` — the search's
held-out slot, which decides which genome to keep, is fed the validation set.

```bash
pip install -e '.[jax]'
python -m mnistbench run records/dmuglich/did --point s --device cuda
```
