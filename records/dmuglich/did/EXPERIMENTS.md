# GA vs DID vs backprop — experiment log (MNIST + CIFAR-10)

Can a **gradient-free search** train a fully-binary LUT network competitively with **backprop**, if we
let it evolve the **wiring** (which two inputs each gate reads) and not just the truth tables?

Scripts: `../evo_algos_mnist.py` (this record — the budget-matched GA / DID / hybrid harness), built
on `records/nherr/mnist-ga/ga_bits_wiring_mnist.py` (the GA net; free / codebook / fixed wiring) and
compared against `records/nherr/mnist-ga/backprop_bits_mnist.py` (STE + soft backprop — same net,
same thermometer pipeline, same metrics). All support `--dataset {mnist,cifar10}`. Data:
`pareto.csv` (the headline comparison; converged cells only), `algo_comparison.csv` (every
budget-matched run), `sweep_metrics.csv` (λ × pop × width sweep), `curves.csv` (first A/B).
`plot_pareto.py` renders `pareto.csv` into `pareto.svg` (`--mnist` → `pareto_mnist.svg`).

## Verdict

**Backprop wins.** The GA is competitive only in a narrow small-model band on MNIST, and loses
decisively on CIFAR-10 — at ~2000× the training FLOPs throughout.

MNIST, accuracy at matched `model_memory_bytes`:

| bytes | backprop | GA (best wiring scheme) | winner |
|---|---|---|---|
| 2,297 | 86.8% | 86.1% (fixed) | backprop |
| 4,596 | 90.1% | **91.4%** (codebook K=4) | **GA +1.3** |
| 5,745 | 91.2% | **91.9%** (codebook K=8) | **GA +0.7** |
| 6,894 | 91.3% | **91.9%** (codebook K=16) | **GA +0.6** |
| 11,490 | **93.5%** | 91.8% (codebook K=256) | backprop +1.8 |
| 16,500 | **94.2%** | 92.1% (free wiring) | backprop +2.2 |

CIFAR-10 — no band, no crossover:

| bytes | backprop | GA | gap |
|---|---|---|---|
| 5,745 | **44.7%** | 38.2% (codebook K=8) | backprop **+6.6** |
| 16,500 | **48.7%** | 39.1% (free, 18 KB) | backprop **+9.6** |

*Later refinements move both frontiers past these tables:* soft-relaxation training (`--soft`,
DiffLogic-style; flat τ beats annealed) lifts backprop to **92.0%** at 5,745 B, the GA's
converged ceiling at 5,745 B is **93.2%** (200k gens), and **rewiring DID** (`--did-rewire`,
below) overtakes both. The finalized, **converged** matched-config comparison is the **Final
leaderboard** directly below — uniform and funnel shapes at five byte classes, label-trained and
teacher-distilled (soft targets from a 99.5% CNN), every cell run out (DID 51.2M evals, backprop
plateau-verified 600 epochs, GA 200k gens). Headlines: label-trained rewire-DID wins every MNIST
cell but 33 KB uniform; teacher-distilled backprop takes over from 16.5 KB up; the CIFAR tables
above are likewise superseded by fresh current-code runs (CIFAR table below) where continuous
training wins every cell — the sign survives, the numbers move. *One further measurement moves
it again:* the converged DID plateaus turn out to be an acceptance artifact, not a ceiling —
confirmed top-k harvesting (see **The plateau is an acceptance artifact**, below) lifts
converged CIFAR DID from 40.51 past backprop's 42.28.

**Why.** At equal *bytes* backprop buys ~2.5× more gates, because its wiring is a seeded random draw —
*structural*, regenerated at load, **0 bytes** — while learned wiring must be stored. And backprop can
*exploit* those gates: it is capacity-limited. The GA is **search**-limited — more gates enlarge the
genome and dilute the search, so it cannot convert capacity into accuracy. On CIFAR, where capacity is
the binding constraint, that asymmetry is fatal.

## Final leaderboard: one circuit, every optimizer

The finalized matched-config comparison. One architecture per byte class — thermometer input,
2-input LUT gates, K=8 codebook wiring, identical genome space, identical deployed artifact —
and only the optimizer differs. Discrete methods get the identical evaluation budget; the
continuous baseline gets its converged 600-epoch recipe and is hardened into the same genome
space. Every row is detailed in the sections below; run-level data in `algo_comparison.csv`.

**5,745 B — [3072, 1024, 500], 4,596 gates** (10.24M evals unless noted; `+ teacher` = distilled
against the 99.5% CNN, α=1, T=4):

| optimizer | test acc | converged | + teacher |
|---|---|---|---|
| **rewire-DID, joint + dedup** | **94.82 ± 0.03** (3 seeds) | — | **95.84 ± 0.06** (3 seeds, best 95.92) |
| rewire-DID | 94.65 ± 0.14 (3 seeds) | 94.87 @ 51.2M | split: proposals-only 95.00 / acceptance-only 95.49 |
| rewire-DID + parent-child | 94.52 | — | — |
| best GA × rewire-DID hybrid | 94.46 | — | — |
| continuous (soft, hardened) | 92.0 | 600 ep | 93.40 |
| GA (pop 512) | 91.70 | 93.2 @ 102.4M | **90.30** — the teacher *hurts* a population |
| DID, frozen wiring | 83.3 | 83.7 @ 51.2M | — |
| `hc` control (random proposals) | 81.5 | — | — |

Deploy (bit-exact pruning): converged rewire-DID **94.87 at 3,050 B**; distilled joint+dedup
**95.92 at 3,049 B** — the frontier point.

**16,500 B — 13,200 gates, same codebook space** (matched 10.24M evals; means over seeds where
shown; the converged numbers live in the sweep table below):

| optimizer | uniform 4400×3 | funnel 8800/2950/1450 | uniform + teacher | funnel + teacher |
|---|---|---|---|---|
| rewire-DID | mean **96.88** (dedup 96.92) | **96.42** | 97.49 ± 0.01 | **96.98 ± 0.07** |
| continuous (soft, hardened) | mean **96.88** (± 0.01) | 95.21 | **97.98 ± 0.16** (best 98.09) | 96.77 ± 0.11 |
| GA | 88.92 | 91.01 | — | — |

Deploy: pruned 16.5 KB uniform genomes → **97.01 at 13,943 B** (no teacher), **97.48 at
14,315 B** (distilled).

**Byte-class sweep — converged** (`pareto.svg`, regenerated by `plot_pareto.py`). Every cell is
run to convergence: rewire-DID at ~51.2M evals (warm-started continuations — every 10.24M run
was still accepting at budget end), backprop at its 600-epoch recipe with the plateau verified
(best epoch 100–325 from 9.2 KB up), teacher = the 99.5% CNN at α=1, T=4. Pruned deploys are
bit-exact:

| class | shape | DID | backprop | DID + teacher | bp + teacher | distilled DID pruned |
|---|---|---|---|---|---|---|
| 2,300 B | [620, 620, 600] | **94.81** | 90.98 | **96.02** | 92.59† | 96.02 @ 2,145 B |
| 2,300 B | funnel [1230, 410, 200] | **91.84** | 87.52 | **92.97** | 87.99 | 92.97 @ 1,348 B |
| 5,745 B | funnel [3072, 1024, 500] | **94.87** | 92.00 | **96.00** | 93.40 | 96.00 @ 3,065 B |
| 9,200 B | [2460, 2450, 2450] | **96.75** | 95.91 | **97.41** | 97.21 | 97.41 @ 8,020 B |
| 9,200 B | funnel [4900, 1640, 820] | **95.83** | 93.70 | **96.65** | 95.24 | 96.65 @ 4,737 B |
| 16,500 B | [4400 × 3] | **97.11** | 96.88 | 97.55 | **97.98** | 97.55 @ 14,487 B |
| 16,500 B | funnel [8800, 2950, 1450] | **96.51** | 95.21 | **97.16** | 96.84 | 97.16 @ 8,219 B |
| 33,000 B | [8800 × 3] | 97.27 | **97.71** | 97.72 | **98.49** | 97.72 @ 29,575 B |
| 33,000 B | funnel [17600, 5900, 2900] | **96.94** | 96.61 | 97.38 | **97.86** | 97.38 @ 16,550 B |

† 1,200 epochs (the 600-epoch 92.46 had not plateaued; every other backprop cell had).

Convergence moved numbers but not the contested verdict: the extra 40.96M evals buy DID +0.05
at 16.5 KB and +0.13 at 33 KB under the teacher — **the high-byte backprop lead is real, not a
budget artifact**. What convergence *did* flip: label-trained 16.5 KB uniform goes from a dead
tie to **DID +0.23**, and label-trained DID now sweeps every funnel class at every size. So the
crossover stands, one class later without a teacher: DID wins every label-trained cell except
33 KB uniform; with a teacher, backprop takes over from 16.5 KB. Funnels prune spectacularly
(~40–50% live vs ~77–85% for uniform), so pruned funnel deploys own the sub-5 KB frontier:
92.97 @ 1,348 B, 96.65 @ 4,737 B.

**CIFAR-10 — same circuits, harder task** (all current-code, label-trained, converged: DID
51.2M evals, backprop 600 epochs, GA 102.4M evals; the legacy fixed-wiring and old-GA rows are
dropped from `pareto.csv`):

| class | shape | rewire-DID | backprop-soft | GA |
|---|---|---|---|---|
| 5,745 B | funnel [3072, 1024, 500] | 40.51 | **42.28** | 39.53 |
| 16,500 B | [4400 × 3] | 44.65 | **49.72** | 36.29 |
| 16,500 B | funnel [8800, 2950, 1450] | 43.68 | **48.21** | 39.06 |

On CIFAR the order inverts: hardened continuous training wins every cell — by five points at
16.5 KB. The MNIST small-class DID edge does not transfer; where the task needs more than
sparse pixel logic, gradient information beats discrete search at every byte class tested.
(DID pruned deploys: 40.51 @ 3,185 B, 43.68 @ 9,083 B, 44.65 @ 15,740 B.)

The shape of the result: a **population is the wrong optimizer at every scale and on both
tasks** — budget-limited at 5,745 B, search-limited at 16,500 B (on MNIST it scores below its
own small-net 91.7 there; on CIFAR its 16.5 KB uniform cell scores below its own 5,745 B one),
exactly neutral when composed with or seeded by the local search, and the only optimizer a
teacher makes *worse*. **Label-trained on MNIST, discrete local search with topology moves
beats hardened continuous training in every cell but 33 KB uniform.** **Teacher-trained, the
byte class decides:** distilled rewire-DID wins through 9.2 KB, distilled backprop from
16.5 KB up. **On CIFAR, continuous wins everywhere** — the discrete edge is a small-model,
easy-task regime. Continuous training also remains two to three orders of magnitude cheaper in
FLOPs (float math vs exact trials over bit circuits; wall-clock gap is far smaller because the
discrete forward is bit-packed) — the discrete premium is what buys the byte-parity wins above.

## What did work: the wiring codebook

Evolving free wiring costs ~13 bits per wire (a full source index) — 7× the 4 bits of a truth table,
which is what made the artifact balloon. Instead: generate **K candidate wirings from a fixed seed**
(structural → 0 bytes) and let each gate evolve only a **choice index** among them (`log2(K)` bits).

| wiring | bits/wire | MNIST | CIFAR-10 |
|---|---|---|---|
| fixed (1 option) | 0 | 86.1% | 36.2% |
| codebook K=4 | 2 | 91.4% | — |
| **codebook K=8** | **3** | **91.9%** | **38.2%** |
| codebook K=16 | 4 | 91.9% | 38.1% |
| codebook K=256 | 8 | 91.8% | — |
| free (any input) | 13 | 92.1% | 39.1% |

**K=8 recovers most of the wiring gain at ~3× smaller an artifact, and it saturates by K=8–16 — on
both datasets.** A gate does not need to find its *optimal* input among thousands; it needs a *handful
of alternatives* to escape the single random assignment it was dealt. Going 1 → 8 options buys +5.8
points (MNIST); going 8 → 5,488 buys +0.2.

Corollary, measured: **selecting a global wiring *seed* is worthless.** Across 8 seeds the accuracy
spread is σ = 0.28% (86.2–87.0%), so best-of-N buys ~+0.4 — random wirings are interchangeable. The
gain comes from *per-gate* freedom, not from a lucky global draw.

## Selection signal

Fitness = `margin + λ · batch_accuracy` (margin = true-class popcount − best distractor). Adding
accuracy is worth **+5.6 points** over margin alone (77.0% → 82.7% at the time it was tested). λ's
optimum **scales with population** (λ≈50 at pop 128; λ≈100–125 at pop 512) and flattens past ~100.
Population is the GA's strongest lever (128 → 512 is worth ~+2), saturating by ~512–1024.

## DID: influence-guided local search vs the GA

`../evo_algos_mnist.py --algo did` runs Discrete Influence Descent in the same harness — same net,
codebook wiring frozen at K=8, same fresh-8k-batch noise profile, every exact trial charged as one
evaluation. Closed-form output influence through the fixed popcount head, **signed**-sensitivity
backpropagation (correction #3), per-gate pattern coefficients `C_{j,p}`, closed-form best-response
tables. Proposals are single-gate table changes with negative surrogate delta, ranked globally; the
top `--did-props` (512) are tried **one at a time** and accepted only if the exact batch loss
drops. `--did-parent-child` adds counterfactual two-gate parent→child bundles to the same ranked
pool. `--algo hc` is the control: the identical acceptance loop fed uniform random proposals.

At the shared 10.24M-evaluation budget (MNIST, 1 GH200):

| algo | test acc | evals to 75% | accepts/sweep at budget end | train_seconds |
|---|---|---|---|---|
| ga | **91.7%** | 4.10M | — | 468 |
| did + parent-child | 82.4% | **≤0.013M** | ~58/512 | 876 |
| did | 81.6% | ≤0.013M | ~165/375 | 1024 |
| hc (random proposals) | 81.5% | 0.09M | ~4/512 | 794 |

Three reads:

1. **Influence-ranked proposals dominate random ones under the same acceptance regime.** Both DID
   variants reach ~80% inside the first 0.1% of the budget (sweep 25!), ~300× fewer evaluations
   than the GA needs and ~7× fewer than hc needs for 75%.
2. **DID's ceiling is ~82, and all of its value arrives in the first 1M evals.** Converged (200k
   sweeps = 102.4M evals): singleton 81.8%, parent-child 82.7% — 10x the budget buys +0.2/+0.3.
   The trajectory tells the story: 80.5% inside 0.1M evals (13 GPU-seconds), 81.6% at 1M, then
   flat — the ~40% end-of-run accept rate is fresh-batch churn, not progress. The GA's converged
   ceiling is 93.2%: exact-acceptance local search saturates ~10 points below population search,
   with or without influence.
3. Parent-child motifs claim ~2/3 of the ranked slots at accept rates comparable to singletons —
   the non-residualized parent score's known ranking bias stays benign after the sign fix too.
4. **The hc gap understates DID's per-trial advantage.** A statically-dead gate (~65% of the
   funnel) has λ ≡ 0, so C ≡ 0 and DID never proposes it — correctly, since flipping it is
   behaviourally null. hc spends ~2/3 of its random trials on exactly these null moves.

A post-fix audit (vs the reference implementation, plus an independent re-derivation of the
surrogate from the multilinear relaxation) found no further gaps; it added three selftest guards —
batch-scale invariance of C, a ranking-calibration check (top-ranked proposals must beat random on
*exact* deltas: the behavioural test the sign bug fails, 0.29 vs 0.57 improvement rate), and the
fixed-batch zero-accept-plateau invariant.

## Where DID has value — and where it doesn't

Can DID's proposal quality lift the GA past 93.2? Every coupling shape was tried at the shared
10.24M budget (plain ga: 91.7, deterministic at seed 0):

| coupling | test acc | vs ga |
|---|---|---|
| memetic — Lamarckian DID bursts on the elite, every 50 gens | 89.4 | **−2.3** |
| memetic, every 500 gens | 89.5 | **−2.2** |
| influence-sampled mutation kernel, T=1 | 90.8 | −0.9 |
| influence-sampled mutation kernel, T=4 | 90.5 | −1.2 |
| DID warm-start (0.1M evals → 80.5%) → GA | 91.6 | −0.1 |
| DID warm-start (1M evals → 81.6%) → GA | 91.5 | −0.2 |
| did with Metropolis acceptance (T0 = 3e-4, annealed) | 80.8 | vs did −0.9 |
| (non-DID lever) pop 1024 × 10000 gens | 90.6 | −1.2 |

- **Mid-run injection is harmful and dose-independent** — 40 bursts cost the same −2.3 as 400.
  The damage is structural, not cumulative: early high-accept bursts commit the population to
  DID's ~82 basin and the GA never escapes back to its own trajectory.
- **Warm starts are neutral.** The GA spends its first ~8k generations *dismantling* the seed
  (test drops below the DID genome before recovering) and finishes where it would have finished
  anyway. DID content simply does not transfer into population search.
- **Polish fails hardest.** From the GA's 91.7 genome, 2M evals of DID drop test to 89.9 —
  within the FIRST sweep — while random-proposal hc under the same acceptance only slips to
  91.2. At the top, strict single-8k-batch CE acceptance is the failure mode: accepted deltas
  are batch noise plus CE-vs-accuracy mismatch, and DID overfits fastest precisely because its
  proposals pass the gate so reliably. (The reference implementation's z-confirmed sub-batch
  acceptance exists for exactly this; out of scope for the core comparison.)
- pop 1024 at equal budget loses 1.2: past 512 the GA needs generations, not population.

**DID's value map.** Dominant in the first ~1M evaluations — 80.5% in 13 GPU-seconds (0.1M
evals), 81.6% at 1M, ~300× cheaper than the GA to that level — then a hard ~82 ceiling, and
zero-to-negative as an ingredient in the GA at any phase. Use DID to get a decent circuit nearly
for free; use the GA for quality in the small-artifact class (93.2 at 5,745 B); use soft-trained
backprop where wiring is free (94.8 at 16,500 B, `--soft`; 92.0 hardened in the GA's own
artifact class — still behind the GA's 93.2). The claimed 96% apples-to-apples backprop did not
reproduce under any recipe tried (best anywhere: 94.8).

Backprop on the **same artifact class** (`--wire-codebook 8`: STE argmax over the same structural
candidates, hardening bit-exactly to a `(tables, wa, wb)` genome — verified against the jax
circuit): **89.4% converged**. With fixed random wiring it converges at 79.7% — correction #1's
"~80% funnel cap" was a *fixed-wiring* cap, not a shape cap: learnable wiring through the same
500-bit readout is worth +9.7. Uniform-width fixed wiring at gate parity: 86.1%. DID's wall-clock
is ~2× the GA's at equal evaluations: its trials run sequentially through a `lax.scan` where the
GA evaluates its whole population in one vmap.

## Second-order DID: curvature and objective-aligned acceptance

DID's first-order surrogate `C¹` is a *linear* Taylor model of the loss in the flip; the natural
next question is whether a curvature term or a better acceptance rule lifts the ~82 ceiling. The
round added four flags (`--did-order2`, `--did-ema`, `--did-confirm`, `--did-accept-fit`) and one
PoC (`poc_did2.py`).

**The second-order model, validated.** Seed a diagonal-GGN curvature `Γ_out = p(1−p)/(G·n)` at the
head (`did_head_gamma`) — the exact diagonal of the head Hessian (selftest 11: matches
`jax.hessian` on the packed-bit loss). Propagate it backward by the **squared** sensitivity, which
for `d ∈ {−1,0,+1}` is exactly the unsigned XOR mask (right for curvature, wrong for the signed
`λ`), into per-pattern bins `C² ≥ 0`. The effective coefficient `C̃ = C¹ + ½(1−2t)C²` drops into
the existing best-response and reproduces the exact quadratic surrogate delta (selftest 12:
brute-force minimality + damping-only subset). So the machinery is correct — it is a diagonal-GGN,
curvature-damped *proposal oracle*, not Newton descent.

**PoC (`poc_did2.py`).** On a small real net across a descent, first-order `C¹`'s in-sample
Spearman vs exact deltas collapses **0.87 → 0.03**; adding curvature (`C̃`) holds it at **~0.98
everywhere**. But at the plateau the gains that curvature ranks correctly in-sample are
**batch-specific** — mean delta on an independent batch ≈ 0. Curvature fixes *which* flips help on
the batch you scored; it cannot make a batch-specific gain generalize.

**Arms at the shared 10.24M budget** (vs plain `did` 81.6):

| arm | test acc | note |
|---|---|---|
| `did2` (curvature-damped proposals) | 81.89 | = plain DID's converged ceiling, at 10× less budget |
| `did_conf` (confirm-batch acceptance) | 81.16 | strict 2-batch gate does not lift |
| `did2_conf` | 81.23 | curvature + confirm, no lift |
| `did2_ema_conf` (EMA over C bins) | 80.40 | EMA always hurts — stale bins misrank a fresh batch |
| **`did2_fit`** (accept on the GA objective) | **83.32** | best standalone DID; beats didpc converged |

Converged (51.2M evals, 100k gens): **`conv_did2` 81.87** — flat from ~26M on, confirming curvature
buys *speed* (it reaches the ceiling ~10× sooner) but not a higher ceiling. **`conv_did2fit`
83.66** — reached at ~10M evals and flat through 51M, so fit-acceptance genuinely raises the
ceiling by ~1.5 points and does so early.

**Polish matrix** (from the GA's 91.7 genome, 2M DID evals):

| acceptance | first sweep | end |
|---|---|---|
| CE (`did2`) | −1.8 instantly | 89.9 |
| CE + curvature + confirm | ≈ flat | 89.9 |
| fit (`--did-accept-fit`) | −0.2 | 90.0 (drifts) |
| fit + confirm | **holds 91.7** first sweep | 90.5 (drifts) |

**The diagnosis: an accept-only ratchet.** Two effects stack at the top. (1) *CE/accuracy
decoupling* — a genuine, confirmed 0.15-nat CE gain cost 1.8 test points, so a CE surrogate
optimizes the wrong thing near the ceiling; `--did-accept-fit` removes exactly this mismatch and is
why it is the only lever that ever raises the DID ceiling. (2) *Multiple testing without a revert
mechanism* — a sweep re-ranks ~70k moves, each strict accept-gate has a finite false-positive rate,
and because acceptance is monotone (a move once accepted is never undone) false positives
accumulate: fit+confirm holds the first sweep but drifts down over many sweeps. **The GA's
population selection is precisely the revert mechanism local search lacks** — a bad child simply
fails to reproduce next generation, so the population never ratchets. This is why polish has no
exploitable value and why DID content does not transfer into population search, and it motivates
treating the *GA's own* selection noise as the next lever (below).

## Selection noise in the GA: measured, then demoted

If acceptance noise ratchets DID, the same batch noise should blunt the *GA's* selection late in a
run. A PoC (`poc_selnoise.py`: mutant populations spread around the converged 91.7 genome, exact
fitness vs 8k-batch fitness) confirms the regime is extreme — at end-of-anneal mutation spread only
4/255 mutants are truly better, the best true gain (+0.0011) sits ~50× below the paired 8k-batch
noise (σ≈0.053), improvement detection is a coin flip (0.49), and the true best lands in the batch
top-4 with p=0.167. Averaging 8 batches fixes the statistics (detection 0.7, false-accept 0.9%).

But every precision lever **loses at the 10.24M budget**, where the run is still improving
(vs `ga` 91.7): batch-annealing `--batch-end` 89.0/90.8 (×32/×16), elite re-evaluation
`--elite-reval 3` 91.0, both combined 90.2, zero-cost LCB history elitism `--elite-hist` 91.4,
and fit-gated memetic bursts 91.3 (up from 89.5 for CE-gated — fit-gating recovers most of the
memetic drag, still no net win). Pre-convergence, evals spent on *precision* are evals not spent
on *search*; history averaging is additionally biased while the fitness distribution is
nonstationary. These levers target the converged plateau only — and the rewiring result below
made that regime obsolete before they were worth converging.

## Rewiring: DID over the codebook topology

Every DID mode above froze the wiring — DID searched the 4,596 truth tables while the GA searched
tables *and* the K=8 codebook choice per wire. That is a confounded comparison, and it left open
whether DID's ~82–84 ceiling was the surrogate or the frozen topology. `--did-rewire` closes the
gap: counterfactual influence bins are computed **per codebook candidate and port**
(`did_rewire_c` — two einsums per port over the cached activations; complementary patterns from
the λ column sums), each candidate wire is scored jointly with its **best-response truth table**
(a rewire proposal never has to keep the table tuned for the old input), and the move delta adds
the base-pattern correction so rewires and table flips rank in one global pool. The trial scan
carries `(code, wa, wb)` so accepted rewires compound within a sweep. Wiring stays inside the
seeded K=8 codebook: the artifact is unchanged — 3 bits per wire, 5,745 B.

Results at 5,745 B (vs `ga` 91.7, GA converged 93.2, soft backprop same-class 92.0):

| arm | evals | test acc | note |
|---|---|---|---|
| **`did_rw`** (CE accept + rewire) | 10.24M | **94.79** | still accepting ~100 moves/sweep (~95% rewires) at budget end — *not converged* |
| `did_rwfit` (fit accept + rewire) | 10.24M | 94.46 | acceptance-rule **inversion**: with topology moves CE beats fit |
| `did_rwj` (+ joint (u,v) K² pairs) | 10.24M | 94.64 | two-port pairs dominate the ranked pool but don't pay at this budget — a wash *(until dedup, below)* |
| `did_pcrw` (+ parent-child motifs) | 10.24M | 94.52 | inside the plain seed spread — table-space bundles add nothing once topology moves exist |
| fit + rewire (diagnostic) | 0.031M | 90.9 | 14 s on one GPU; ~300× less compute than the GA needs to reach this level |
| `ga_memrw500` (GA + rewire-DID bursts) | 10.24M | 91.06 | Lamarckian hybrid loses even with rewiring in the bursts |
| `ws_rwfit` (GA warm-started from `did_rwfit`) | +10.24M | 94.46 | the GA ends **exactly at its seed's accuracy** — zero added |

So the ceiling was the **frozen wiring, not the surrogate**: given topology moves, the same
influence machinery that plateaued at 82 beats everything else in the class inside the standard
budget. The acceptance inversion makes sense in the ratchet frame: rewires are large, real moves
on which CE and accuracy agree, so the CE gate's sharper signal wins where table-flip polish lost
to CE/accuracy decoupling. And the hybrid verdict is now complete: bursts hurt (91.06), a
warm-started GA is exactly neutral (94.46 in, 94.46 out — the same neutrality ws100k/ws1m showed),
while simply *continuing* rewire-DID for the same additional evals gains another ~0.3. **A
population adds nothing on top of rewire-DID; the GA is retired as the frontier.**

**Converged (51.2M evals):** `conv_did_rw` **94.84**, `conv_did_rwfit` **94.85**, with dedup
(`conv_did_rw_dd`, next section) **94.87** — the acceptance rules converge to the same ceiling
(CE is just faster getting there), with no drift-down: at rewire scale the accept-only ratchet
never bites. The converged rewire-DID
ceiling beats the GA's converged 93.2 by +1.65 at a fifth of the evaluations. Deploy-time
pruning (bit-exact, `prune_deploy_mnist.py --genome`) takes the converged dedup genome to
**94.87 at 3,050 B** (43.1% live) and the 16.5 KB uniform genome to 97.01 at 13,943 B (74.5% live) —
rewire-DID circuits are markedly more live than the GA's ~35%: topology moves recruit gates
into sensitive paths instead of leaving them dead.

**Same architecture, larger class (16,500 B = 13,200 gates).** The apples-to-apples scale test:
identical codebook architecture, only the optimizer differs — rewire-DID vs the soft relaxation
hardened into the same genome space (`--soft --tau1 1.0 --wire-codebook 8`), seeds 0/1/2:

| architecture | rewire-DID (10.24M evals) | soft backprop (600 ep) | GA (10.24M evals) |
|---|---|---|---|
| uniform 4400×3 | 97.01 / 96.82 / 96.81 (mean **96.88**) | 96.87 / 96.88 / 96.89 (mean **96.88**) | 88.92 |
| funnel 8800/2950/1450 | **96.42** (seed 0) | 95.21 (seed 0) | 91.01 |

(For scale, `did_rw` at 5,745 B across the same seeds: 94.79 / 94.51 / 94.66, mean 94.65 ± 0.14.)

On the uniform shape — backprop's best — the two optimizers land in a **dead tie** at the mean,
with backprop eerily seed-stable (±0.01) and rewire-DID spread ±0.11. On the funnel shape the
+1.2 margin is far outside either spread: evolution exploits the funnel that backprop can't (the
same shape asymmetry the fixed-wiring runs showed, now at equal wiring freedom). So at equal
bytes *and* equal architecture, rewire-DID is never behind continuous training, and ahead where
the architecture is search-friendly — at the size class where backprop's advantage was supposed
to live (its old fixed-wiring best here was 94.8). The GA column quantifies its
search-limitation directly: at 3× the gates it scores *below* its own 91.7 on the small net —
population search cannot convert capacity into accuracy, while both DID and backprop convert
the same gates into ~+2 points.

*Ops footnote:* the fit+rewire kernel is large enough that `ptxas` spills temp files; on nodes
where the per-job `TMPDIR` is unwritable this crashes with misleading segfaults — export
`TMPDIR=/tmp` in the job script.

## Proposal ranking: calibration is fine; duplication was the waste

Rewire-DID ranks a pool of ~70k proposals per sweep on one cached linearisation and exact-trials
the top 512. The obvious suspects for lost ground are **over-optimism** (winner's curse at the top
of a noisy ranking) and **mis-ordering** across proposal types with different error scales. A
measurement pass (`poc_rankcal.py`: the real net at three descent stages — sweep 50, sweep 500,
converged — all four proposal types, each top proposal re-scored by an exact *applied-alone*
trial) exonerated both:

- **Over-optimism is real but harmless.** Realized deltas run 0.3–0.6× predicted (joint pairs
  ~10× optimistic early), yet per-type precision@top — the fraction of trialed proposals that
  genuinely improve the loss — is already **80–90% at every stage** (Spearman ρ 0.5–0.8), and
  per-type linear recalibration buys no consistent precision. Acceptance is by exact trial, so
  inflated magnitudes cost nothing once the ordering is right. Isotonic recalibration, shrinkage,
  conformal gating and extreme-value corrections were dropped on this evidence.
- **The waste is duplication.** 403–456 of the raw top 512 target a gate that a better-ranked
  proposal also writes. Trialed sequentially, each accepted winner stales the cached linearisation
  of everything behind it on the same gate: applied *alone*, ~440 of the top 512 improve the loss;
  applied in ranked sequence, ~50 are accepted.

`--did-dedup` therefore walks the full ranking and keeps only the **first proposal per written
gate** — 512 trials on 512 distinct gates. Accepts jump ~50 → ~200 per sweep (91.3% inside 25
sweeps / 10 GPU-s, vs 87.6 without). At budget:

| arm | evals | test acc | read |
|---|---|---|---|
| `did_rw_dd` | 10.24M | 94.65 | equals the plain-rewire seed mean — same destination, reached far earlier |
| **`did_rwj_dd`** | 10.24M | **94.86 / 94.80 / 94.80** (mean **94.82 ± 0.03**) | joint K² pairs flip from a wash (94.64) to the best 5,745 B number — at the 51.2M singleton ceiling in a fifth of the evals, and tighter across seeds (± 0.03 vs ± 0.14) |
| `did_rw16u_dd` | 10.24M | 96.92 | within the 16.5 KB seed spread (96.81–97.01) |
| `conv_did_rw_dd` | 51.2M | **94.87** | the converged ceiling nudges up (94.84/94.85 without dedup) — and joint+dedup reaches it at a fifth of the evals |

The joint result is the one that matters: without dedup the K² pairs dominate the *ranking* but
their trials are spent on hundreds of variants of the same few gates, so breadth never happens;
with one shot per gate, the two-port moves' larger per-move gains actually land. Parent-child
motifs tell the complementary story (`did_pcrw` above, 94.52): once topology moves exist, a
second table-space move type has nothing left to add — the productive axis of composition is
*wider moves per gate*, not *more gates per move*.

## Distillation: a 99.5% teacher as the target

Swap the one-hot labels for a CNN teacher's softened logits — `--distill`, targets
`(1−α)·onehot + α·softmax(teacher/T)` — and every optimizer trains on the same soft-target CE
(`teacher_mnist.py` trains the teacher: 99.52% test, logits row-aligned and fingerprinted). For
DID the swap is one line in the head: the λ seed becomes `softmax − t`, and the whole
influence/C/rewire machinery is target-agnostic (selftests 18–19 pin the soft path to
multilinear-autodiff ground truth). The mechanistic bet was that this is *more than variance
reduction*: at 97% accuracy the hard-CE λ is near-zero on every confidently-correct sample, so
the C bins that rank proposals are effectively estimated from the ~250/8,000 still-misclassified
samples per batch; soft targets keep all 8,000 informative — a proposal-quality effect, not just
a smoother acceptance gate.

Temperature smoke (0.1M evals, joint+dedup): hard 93.15 · T=1 93.11 · T=2 93.28 · **T=4 94.13**
· T=6 93.68 · T=8 92.79; mixing labels back (α=0.7) only dilutes → **α=1, T=4 everywhere**.

At the full budget, one teacher and one target form across all three optimizer families:

| arm | class | test acc | vs no teacher |
|---|---|---|---|
| **`did_rwj_dd_dist`** (rewire-DID joint+dedup, 3 seeds) | 5,745 B | **95.84 ± 0.06** | 94.82 ± 0.03 → **+1.02** |
| `did_dist_prop` (teacher in proposals only) | 5,745 B | 95.00 | +0.18 |
| `did_dist_acc` (teacher in acceptance only) | 5,745 B | 95.49 | +0.67 |
| `ga_dist` (GA selecting on −softCE) | 5,745 B | 90.30 | 91.70 → **−1.40** |
| `bp_soft_cb8_dist` (hardened continuous) | 5,745 B | 93.40 | 92.0 → +1.40 |
| `did_rw16u_dd_dist` | 16.5 KB u | 97.38 | 96.92 → +0.46 |
| `did_rwj16u_dd` (joint, no teacher) | 16.5 KB u | 96.92 | joint alone is neutral at this scale |
| **`did_rwj16u_dd_dist`** (3 seeds) | 16.5 KB u | **97.49 ± 0.01** | 96.92 → +0.57 |
| **`bp_soft_cb8_16u_dist`** (3 seeds) | 16.5 KB u | **97.98 ± 0.16** | 96.88 → **+1.10** |
| `did_rwj16f_dd` (funnel, no teacher) | 16.5 KB f | 96.28 | ≈ plain rewire's 96.42 |
| **`did_rwj16f_dd_dist`** (3 seeds) | 16.5 KB f | **96.98 ± 0.07** | +0.70 |
| `bp_soft_cb8_16f_dist` (2 seeds) | 16.5 KB f | 96.77 ± 0.11 | 95.21 → **+1.56** |

Three findings. **First, the mechanism split answers the question that motivated this:** the
teacher helps through *both* channels — proposals alone +0.18 (six seed-sigmas: the denser λ
genuinely surfaces better gates), acceptance alone +0.67 (the smoother, generalizing gate —
consistent with acceptance having been the historical bottleneck), and together +1.02 ± 0.06,
*super-additive*: proposing and accepting on the same objective compounds. So yes — more than
variance reduction, on the proposal side specifically. **Second, the teacher inverts with the
optimizer:** rewire-DID converts it into the largest single gain any lever has produced at
5,745 B, hardened continuous training converts it even harder (+1.4/+1.1/+1.6 across cells),
and the GA gets *worse* — a population selecting on soft-CE loses the margin+accuracy signal it
actually needs, the same CE/fitness mismatch every GA coupling has shown. **Third, the teacher
moves the crossover point:** at 5,745 B distilled DID leads distilled backprop by +2.4; at
16.5 KB uniform, distilled backprop (97.98 ± 0.16 — the teacher also costs backprop its ±0.01
seed stability) clears distilled DID (97.49 ± 0.01, itself the most seed-stable arm ever run
here); and the funnel — evolution's +1.2 signature win — compresses to a still-real +0.21
(96.98 ± 0.07 vs 96.77 ± 0.11, three seeds vs two). Without a teacher, rewire-DID was never
behind at equal bytes; with one, the byte class decides the winner: DID small and on the
search-friendly shape, backprop on uniform-at-scale. One-off
teacher cost ≈ 1e14 FLOPs (twelve CNN epochs), amortized across every distilled run and
excluded from per-run `train_flops`.

Deploy-time pruning stays free on distilled genomes (bit-exact, predictions identical): the
5,745 B genome prunes to **95.92 at 3,049 B** (43.1% live — dominating the previous best-bytes
point, 94.87 at 3,050 B), the 16.5 KB uniform genome to **97.48 at 14,315 B** (76.8% live —
distilled circuits are the livest yet).

## The plateau is an acceptance artifact: true-delta anatomy of a converged pool

Every converged DID cell above shares one signature: flat loss with a *healthy accept rate* —
~100 accepts/sweep at ~4e-4 nats each on the acceptance batch, forever. Two readings were on the
table: (A) the single/pair move set is **exhausted** and the accepts are pure winner's curse
(best-512-of-70k selection under batch noise), or (B) a thin tail of true improvements exists but
is **buried** under the noise. `poc_truedelta.py` measured it directly on the converged genomes:
build one production sweep's dedup'd top-512 pool, apply each proposal ALONE, and evaluate it on 16
fresh 8k batches **paired** against the base loss on the same batch. Pairing cancels common-mode
noise — only samples inside the move's cone contribute — so the per-move sigma is ~3.5e-4 nats
(CIFAR) / 7.5e-5 (MNIST) and 16 batches resolve deltas near 1e-4. (Sanity anchor: on a random-init
net the same harness certifies 53/64 proposals as true improvements and a sweep as −0.31 nats.)

**Both readings are wrong.** The true tail is fat *and* detectable — and the production sweep
destroys it:

| converged genome | true improvers (of 512) | best single move | one production sweep, re-measured on fresh batches |
|---|---|---|---|
| CIFAR 5,745 B (40.51%) | **~300** at t<−2 (chance ~16); 238 at t<−4 | −2.5e-3 nats (t −23) | **+0.005 nats** (worse), test 40.5 → 38.8 |
| MNIST 5,745 B (94.87%) | ~97 at t<−2; ~54 at t<−4 | −3.2e-4 nats (t −18) | +0.002 nats, test ~flat |

Selection is not the failure: the moves the sweep accepts are individually good (their true
singles sum to **−0.024** nats on CIFAR). **Composition is.** Applying the top-k true movers
jointly from base and comparing the composite against the sum of its parts (CIFAR, pool 0):

| k | composite (true) | sum of singles | interference gap |
|---|---|---|---|
| 1 | −0.0022 | −0.0022 | 0 (by construction) |
| 4 | −0.0031 | −0.0072 | +0.004 |
| 16 | +0.0016 | −0.0232 | +0.025 |
| 64 | **+0.0973** | −0.0689 | +0.166 (test → 33.7%) |

On MNIST the same probe stays near-additive through k=64 (−0.005 composite, gap +0.002) — which
is why its production sweeps merely tread water while CIFAR's actively damage. "Converged" is a
**churn equilibrium**: each sweep's ~100 same-batch accepts do real harm ≈ real good, and the
best-of-run tracker just remembers the high-water mark.

### Harvest: confirmed top-k acceptance breaks the ceiling

The k-probe prescribes the fix directly: accept *few* moves per sweep, and confirm them on fresh
data. Harvest mode (`--harvest-sweeps`) does exactly that — per sweep: production pool → dedup →
paired confirmation of every candidate on m=8 fresh batches → accept only the top-k with t<−2
(k=4 CIFAR, 16 MNIST), applied jointly. Descending **from the converged genomes**:

- **CIFAR 5,745 B: 40.51 → 44.34** (probe loss 1.711 → 1.542, three chained runs / 22.2M evals).
  It passes backprop's converged 42.28 within ~1.5M evals — 3% of the production budget — and
  the well keeps refilling for ~15M more before accepts thin toward dry. The CIFAR verdict
  flips: the gap was never capacity, it was harvest mechanics.
- **MNIST 5,745 B: 94.87 → 95.18** best (probe 0.281 → 0.266, 7.4M evals); gains land in the
  first ~100 sweeps, then the well dries — the label-trained ceiling was mildly understated,
  not broken.

Harvest evals are charged honestly (n·m + m per sweep ≈ 8× a production sweep), but at a plateau
throughput is worthless, so the trade is free where it matters. The harvested genomes
(`harvest_*_conv*.npz`, production format) are rewire-DID **plus a polish phase**, so they are
logged in `algo_comparison.csv` but not folded into `pareto.csv` yet — that reframing (new method
row vs updated DID row) is a maintainer call.

### Interference is dense, destructive, head-mediated — and a pairwise model tames it

The composition failure raises the question the acceptance work starts from: is interference
*structured* enough to select around? `--pair-movers` measures it. Take the top-32 confirmed
movers (t<−2 over m=8 paired batches, ranked by mean delta — harvest's own acceptance ranking),
apply all 496 pairs jointly from base on the same batches, and decompose I_ij = d_ij − d_i − d_j
per batch. The probe is self-validating: sample-disjoint pairs must compose exactly and do
(|I| ≤ 2e-8), and wherever the second-order model Σd_i + ΣI_ij favors a selection, its prediction
matches the measured composite to ≤1e-4 nats.

| converged genome | CIFAR (40.51) | MNIST (94.87) |
|---|---|---|
| significant movers (of 512) | 313 | 97 |
| mover layer split [L0,L1,L2] | [9, 8, 15] — last layer 4× overrepresented | [19, 8, 5] — ∝ layer size |
| realized cone: gates / samples (p50) | 2 / **5108 of 8000** | 3 / 1008 |
| ΣI over 496 pairs (Σ singles) | **+0.122** (−0.041) | +0.0013 (−0.005) |
| destructive / synergistic pairs (chance ~21 each) | 361 / 133 | 217 / 152 |
| top-10 pairs' share of Σ\|I\| | 0.12 (diffuse) | 0.43 |
| spearman \|I\| vs head-group / sample / gate-cone overlap | +.58 / +.50 / +.17 | +.44 / +.35 / +.10 |

Interference on CIFAR is **dense and diffuse** — no sparse conflict graph to prune — and it is
**head-mediated, not cone-mediated**: a move touches 2–3 gates yet flips head bits on ~64% of
samples, and pairs whose touched head groups collide carry ~17× the interference of pairs that
don't (mean I +6.7e-4 vs +3.9e-5). That is also the CIFAR/MNIST asymmetry in one line: CIFAR's
movers crowd the last layer and each spans most of the batch, so any two accepts collide in the
head; MNIST's movers sit up front with 8× smaller sample cones. Selection races, one sweep from
the converged CIFAR genome (TRUE = composite on the same m=8 paired batches; 4 held-out fresh
batches agree everywhere):

| selection | k | Σ singles | 2nd-order pred | measured TRUE | test 40.51 → |
|---|---|---|---|---|---|
| naive top-16 (harvest ranking) | 16 | −0.024 | +0.029 | +0.013 | 38.7 |
| naive top-32 | 32 | −0.041 | +0.081 | +0.047 | 35.4 |
| gate-cone-disjoint | 16 | −0.022 | +0.025 | +0.035 | 36.1 |
| head-group-disjoint | 4 | −0.0067 | −0.00646 | −0.00646 | 40.4 |
| **greedy on measured I** | **10 (self-chosen)** | −0.0136 | −0.01054 | **−0.01050** | **40.8** |

Three punchlines. (1) Gate-cone disjointness — the intuitive conflict rule — is *worse than
naive*: the interference it dodges was never the problem. (2) Greedy selection on the measured
pairwise model takes ~3× harvest-k4's per-sweep haul, its prediction is exact to 4e-5, and its
natural stopping point **is** the adaptive k: the same rule stops at 10 on CIFAR and 29 on MNIST,
reproducing the k-probe asymmetry with no knob. (3) The model is cheap to feed: restricting pair
measurement to head-colliding pairs (163/496) reproduces the full greedy selection exactly, and a
feature-only linear Î (sample × head-group overlap, zero pair forwards, R² 0.84) still captures
92% of the value — a two-stage sweep (screen → confirm top-32 → head-colliding pairs → greedy)
costs ~2–3× a production sweep. The same probe on the *harvested* genomes finds 2 significant
movers each (their pair additive): the harvest ceilings are genuine single+pair move exhaustion at
this sensitivity, so further gains there need new move classes, not better acceptance.

## Dead gates are inert, not a reservoir

~65% of gates are dead (no statically-sensitive path reaches an output). The *scratch-space*
hypothesis was that mutations accumulate in them at no fitness cost, so a later rewire pulls in a
ready-made variant rather than junk — which would make deadness useful to the search. Wiping dead
gates every 500 gens (`--sterilize-every 500`, wiping is behaviourally free by static liveness) costs
nothing: **91.58% (`zero`) / 91.33% (`rand`) vs 91.70% control**, both inside the ±0.7 noise band the
crossover-variant runs establish. Neither the evolved content (`rand` keeps variation, wipes content)
nor the variation itself (`zero` removes both) matters. Combined with the autopsy (dead gates don't
hurt: the funnel beats the all-live uniform net), the read is that two thirds of the circuit is simply
surplus — which is also why deploy-time pruning is free.

*Caveat:* 500 gens between wipes leaves time for content to re-accumulate; a reservoir on a much
shorter timescale would survive this test.

## Three corrections (recorded so they are not repeated)

1. **The 500-bit readout bottleneck.** The original net funnels `3072 → 1024 → 500`, and the GroupSum
   head splits the *last* layer into 10 classes → only **50 bits per class**. Every early backprop run
   inherited this shape, capping it at ~80% *regardless of width* — widening layers 1–2 while pinning
   the readout at 500 cannot help. With uniform width (`11000×3`, same gate count, same FLOPs) backprop
   jumps to **94.2%**. This briefly produced a bogus "GA beats backprop by +11.7" headline. **Always
   check the readout width before concluding a method has a ceiling.**
2. **"Width is a dead lever" was retracted, then reinstated.** The original width sweep kept the pinned
   500 readout, so it was structurally incapable of showing a width benefit — a real confound. But the
   fixed re-runs show width genuinely *does* hurt the wiring-GA (92.1% at 500 → 87.9% at 8000 readout,
   monotonically), because the GA is search-limited. The conclusion was right; the original evidence
   for it was not.
3. **Unsigned influence propagation (DID's XOR-mask bug).** DID's backward propagated λ with the
   unsigned sensitivity mask `h(1,b) XOR h(0,b)`. λ is a directional gradient: propagation needs the
   signed partial `h(1,b) − h(0,b)`, and the magnitude form silently destroys the sign through every
   inverting path — corrupting C for every layer but the last (~89% of gates) while leaving early
   descent plausible-looking. Exposed only when `hc` (random proposals, same acceptance) overtook DID
   late; pinned by ground-truthing C for ALL layers against `jax.grad` of the circuit's multilinear
   relaxation (selftest 3b — the unsigned form fails it, the signed form passes). With the fix DID's
   apparent 0.79 "plateau" vanished (budget-end accept rate 0.6% → 46%). **Never validate a backward
   pass only against a reference loop that shares its convention — the original selftest did, and
   passed on the bug. Autodiff of the multilinear relaxation is the ground truth.**

## Cost metrics

Both scripts print a `METRICS {...}` line (competition currency, CLAUDE.md Scoring):
`model_memory_bytes` (tables at 4 bits/gate + *stored* wiring; structural wiring costs nothing),
`gate_evaluations` / `train_flops` (GA: `pop·batch·gens·n_gates`, forward-only — the `pop` factor is
the GA's tax; backprop: `3 × samples_seen · n_gates` for fwd+bwd, no population factor),
`samples_seen`, `train_seconds`, `gpu_hours`.

The GA runs at ~3.8e14 FLOPs against backprop's ~2e11 — **~2000×**. Wall-clock is only ~8× apart
(~480 s vs ~60 s), because the GA's ops are bitwise on packed bits (8 samples/byte) while backprop
does float32 math — so gate-eval counts overstate the GA's real cost.
