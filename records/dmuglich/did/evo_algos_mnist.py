"""Budget-matched evolutionary algorithms for a binary LUT network.

Hold the representation, mutation operators, fitness, and evaluation budget fixed, and change only
the search algorithm. The circuit, genome (truth tables + wiring), and mutation are imported from
`ga_bits_wiring_mnist.py`.

Algorithms (`--algo`):
  ga         the incumbent: tournament selection + gate-uniform crossover + elitism.
  aging      Aging (regularized) Evolution — tournament -> mutate -> append child, evict the
             OLDEST, so one early winner cannot own the population forever. Mutation-only.
  mapelites  MAP-Elites — a 2D archive of niches; keep the best genome per niche and breed from
             the archive. Descriptors: (prediction entropy, output activation rate).
  nslc       Novelty Search with Local Competition — reward behavioural novelty AND beating your
             behavioural neighbours, so immature-but-different circuits survive as stepping stones.
  eda        Univariate EDA (UMDA/PBIL) — a Bernoulli per table bit, a Categorical(K) per wire;
             sample, then refit the marginals to the elites.
  snes       Separable Natural Evolution Strategies over continuous latents of the genome.
  did        Discrete Influence Descent — ONE network, no population. Per sweep (one fresh
             batch): closed-form output influence, signed-sensitivity backprop, per-gate
             per-pattern C coefficients. Proposals flip only bits a nonzero C drives, are ranked
             globally by surrogate delta, and the top `did_props` are tried ONE AT A TIME by
             exact forward, accepted only on a strict loss drop. The popcount head is fixed.
  hc         Random hill climber — DID's acceptance regime with uniform random proposals: the
             control that separates proposal quality from the acceptance regime.

DID variants and couplings (flags, not algos):
  --did-parent-child  counterfactual parent-child motifs join the ranked pool: for an edge j->k,
                      each of the 16 parent tables is rolled out through cached activations, the
                      child's best response is closed-form, and (h_j, h_k*) is ONE two-gate
                      proposal scored delta_parent + delta_child|h_j.
  --did-rewire        codebook topology search: counterfactual C bins for every candidate wire
                      per port (the other port keeps its source), each scored jointly with its
                      best-response table; rewires and table singletons rank in one global pool
                      and run through a trial scan that carries the wiring. Composes with
                      --did-accept-fit and --did-parent-child.
  --did-joint         with --did-rewire: also score all K^2 joint (u, v) candidate pairs per
                      gate — two-port moves the port-separable bins cannot see.
  --did-dedup         trial only the best-ranked proposal per written gate: the raw top-512 is
                      ~85% proposals whose gate a better-ranked entry already claims, and those
                      burn their exact trial on a stale genome once the better one accepts.
  --did-order2        curvature-damped proposals: a diagonal-GGN curvature is seeded at the head
                      (p(1-p)/(group*n)), propagated by the squared sensitivity, binned into
                      C2 >= 0, and folded into C~ = C1 + (1-2t)/2 * C2 — a row flips only when
                      its first-order gain beats the curvature penalty.
  --did-ema B         EMA of the C bins across sweeps; acceptance stays exact on the current
                      batch. 0 = off.
  --did-confirm       accepts must strictly improve BOTH the proposal batch and an independent
                      confirm batch (2 evals/trial).
  --did-accept-t0 T0  Metropolis acceptance, T annealed T0 -> T0/1000 over the budget. 0 = exact.
  --did-accept-fit    acceptance gates on the GA's own objective (margin + acc_weight*accuracy)
                      instead of CE; proposals still come from the CE surrogate.
  --distill P         teacher distillation: P is a logits npz from teacher_mnist.py; the one-hot
                      rows in the head loss and the lambda seed become
                      (1-alpha)*onehot + alpha*softmax(teacher/T)  (--distill-alpha,
                      --distill-temp). Every sample then carries a full 10-way constraint, where
                      hard CE concentrates all late-stage signal in the few percent of samples
                      still misclassified. --distill-propose-only / --distill-accept-only split
                      the mechanism: teacher targets in the proposal linearisation only, or in
                      the acceptance objective only. For population algos (ga, ...) the shared
                      fitness becomes -softCE on the same targets — every optimizer selects on
                      the same objective (backprop_bits_mnist.py grew the matching flags).
  --memetic-every N   ga: every N generations run one Lamarckian DID sweep on the current best
                      genome and reinsert it. Burst trials are charged to the budget.
  --mut-influence T   ga: influence-SAMPLED table mutation — mutated gates draw their new table
                      from p_g(t) ~ exp(-C_g . t / (T * scale_g)) instead of flipping random
                      bits. A sampling prior, never a hard filter.
  --save-genome P     write the best genome found to P (.npz: tab, wa, wb) after the run.
  --init-genome P     start from a saved genome: ga seeds half the population with mutated
                      copies (slot 0 exact); did/hc start from the genome itself.

Every algorithm spends the SAME number of network evaluations: `pop * gens`. Local-search modes
charge every acceptance trial (plus a per-sweep scoring surcharge) against that budget; GA modes
with variable per-generation cost (--batch-end, --elite-reval) key their anneal schedules to
budget progress so the run ends exactly at spend.

Run:
    uv run --no-project --with jax --with tyro python evo_algos_mnist.py --algo ga
    uv run --no-project --with jax --with tyro python evo_algos_mnist.py --selftest
"""

from __future__ import annotations

import functools
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import numpy as np
import tyro
from ga_bits_wiring_mnist import Net, load_data, pack


@dataclass
class Config:
    """Shared settings — identical across algorithms so only the search differs."""

    algo: str = "ga"  # ga | aging | mapelites | nslc | eda | snes | gomea | did | hc
    dataset: str = "mnist"  # mnist | cifar10
    widths: list[int] = field(default_factory=lambda: [3072, 1024, 500])
    thresholds: list[int] = field(default_factory=lambda: [32, 64, 96, 128, 160, 192, 224])
    wire_codebook: int = 8  # K candidate wirings per gate (the cheap-to-store scheme)
    pop: int = 512  # population / batch of children per iteration
    gens: int = 20000  # iterations -> total evaluations = pop * gens for every algo
    batch: int = 8000  # samples per fitness evaluation
    mut: float = 0.005  # table bit-flip rate (annealed to mut_end)
    mut_end: float = 0.0004
    wire_mut: float = 0.003  # per-gate rewire rate (annealed to wire_mut_end)
    wire_mut_end: float = 0.0002
    acc_weight: float = 100.0  # fitness = margin + acc_weight * batch_accuracy
    tsize: int = 5  # tournament size (ga, aging)
    use_crossover: bool = True  # ga only: gate-wise crossover on/off (the key ablation)
    bias_beta: float = 0.0  # ga only: importance-biased crossover strength (0 = fair coin)
    bias_every: int = 500  # ga only: gens between sensitivity refreshes
    cone: bool = False  # ga only: cone crossover — parentage propagates backwards along the wiring
    sterilize_every: int = 0  # ga only: every N gens wipe dead gates' tables (0 = off); wiping
    #   is behaviourally free by static liveness, so any accuracy change isolates their content
    sterilize_mode: str = "zero"  # "zero" (constant gates) or "rand" (fresh random tables)
    gomea_trials: int = 19  # gomea: fitness-gated donor-copy trials per generation
    gomea_rebuild: int = 50  # gomea: gens between linkage-tree rebuilds
    gomea_max_subset: int = 512  # gomea: largest linked subset used for mixing
    elite: int = 4  # elitism (ga only)
    grid: int = 16  # MAP-Elites archive is grid x grid niches
    knn: int = 15  # neighbours for novelty / local competition (nslc)
    novelty_w: float = 1.0  # weight on novelty vs local competition (nslc)
    probe: int = 512  # probe samples defining the behaviour descriptor (nslc)
    eda_elite: float = 0.25  # fraction of the population refitted from (eda)
    eda_lr: float = 0.3  # how far the distribution moves toward the elites each round (eda)
    eda_pmin: float = 0.02  # clamp on the marginals — stops a bit locking at 0/1 forever (eda)
    snes_lr_mu: float = 1.0  # SNES step size on the mean (the standard default)
    did_props: int = 512  # did: exact-acceptance trials per sweep, taken in global priority order
    did_parent_child: bool = False  # did: add counterfactual parent-child (2-gate) motif proposals
    did_accept_t0: float = (
        0.0  # did: Metropolis acceptance temperature, annealed to t0/1000 (0 = exact)
    )
    did_order2: bool = False  # did: curvature-damped (diagonal-GGN) proposal scoring
    did_ema: float = 0.0  # did: EMA factor over the C bins across sweeps (0 = single batch)
    did_confirm: bool = False  # did: accepts must also improve an independent batch (2x cost)
    did_accept_fit: bool = False  # did: accept on the GA objective margin + acc_weight*acc, not CE
    did_rewire: bool = False  # did: add codebook rewire proposals (per-port counterfactual
    #   C bins, joint best-response table) to the singleton pool. Composes with did_accept_fit only
    did_joint: bool = False  # did: also score all K^2 joint (u, v) candidate pairs per gate —
    #   two-port rewires the port-separable bins cannot see; requires did_rewire
    did_dedup: bool = False  # did: trial only the best-ranked proposal per written gate —
    #   measured: ~85% of the raw top-512 target an already-claimed gate and burn their trial
    #   on a stale proposal
    distill: str = ""  # did/ga: teacher-logit npz (from teacher_mnist.py). Soft-target CE:
    #   targets = (1-alpha)*onehot + alpha*softmax(teacher/T) replace the one-hot rows in the
    #   DID head loss AND the lambda seed — every sample then constrains all logits, not just
    #   the few percent the circuit misclassifies. For population algos the shared fitness
    #   becomes -softCE on the same targets, so every optimizer selects on the same objective
    distill_alpha: float = 1.0  # did: weight on the teacher's soft target vs the one-hot label
    distill_temp: float = 4.0  # did: teacher softmax temperature
    distill_propose_only: bool = False  # did: teacher shapes lambda/proposals only, acceptance
    #   stays hard-CE (mechanism split: does the teacher sharpen PROPOSALS?)
    distill_accept_only: bool = False  # did: teacher gates acceptance only, proposals stay
    #   hard-CE (mechanism split: is it just a smoother acceptance objective?)
    batch_end: int = 0  # ga: anneal the effective fitness batch to this size (power-of-2
    #   multiple of batch) over the second half of the budget, as averaged fresh base batches;
    #   charged pro-rata, so the run ends when the budget does
    elite_reval: int = 0  # ga: extra independent batches averaged over the top-16 before elitism
    #   (selection-noise guard: tournaments stay single-batch, only the elite slots + best-genome
    #   tracking use the refined estimate; charged 16 evals per extra batch)
    elite_hist: float = 0.0  # ga: LCB elitism z (0 = off). Every generation is an independent
    #   fresh batch, so a surviving elite accumulates free re-evaluations; rank the elite slots by
    #   history mean of CENTERED fitness (batch common mode removed) minus z * sd/sqrt(n) — a
    #   noisy one-batch challenger must clear a significance bar to displace a well-measured
    #   incumbent. Zero extra evaluations.
    memetic_every: int = 0  # ga: gens between Lamarckian DID bursts on the current best (0 = off)
    mut_influence: float = 0.0  # ga: influence-sampled mutation temperature (0 = blind bit flips)
    save_genome: str = ""  # write the best genome (.npz: tab, wa, wb) here after the run
    init_genome: str = ""  # start from this saved genome (ga: seed half the population)
    selftest: bool = False  # run the DID unit checks on a tiny synthetic net and exit
    seed: int = 0
    log_every: int = 2000
    run_name: str = ""
    metrics_out: str = ""


# --------------------------------------------------------------------------------------
# Shared genome ops — IDENTICAL for every algorithm, so only the search differs
# --------------------------------------------------------------------------------------
def mutate(key, net: Net, tab, wa, wb, mut: float, wmut: float):
    """Table bit-flips + per-gate rewires. The one mutation operator, shared by all algorithms."""
    km, kwa, kwb, kma, kmb = jax.random.split(key, 5)
    tab = tab ^ (jax.random.uniform(km, tab.shape) < mut).astype(jnp.uint8)
    newa = (jax.random.uniform(kwa, wa.shape) * net.wire_max).astype(jnp.int32)
    newb = (jax.random.uniform(kwb, wb.shape) * net.wire_max).astype(jnp.int32)
    wa = jnp.where(jax.random.uniform(kma, wa.shape) < wmut, newa, wa)
    wb = jnp.where(jax.random.uniform(kmb, wb.shape) < wmut, newb, wb)
    return tab, wa, wb


def crossover(key, net: Net, tab, wa, wb):
    """Gate-level uniform crossover: a gate's table AND its wires travel together (ga only)."""
    p1t, p2t = tab[0::2], tab[1::2]
    p1a, p2a, p1b, p2b = wa[0::2], wa[1::2], wb[0::2], wb[1::2]
    gmask = jax.random.bernoulli(key, 0.5, (p1t.shape[0], net.n_gates))
    ct = jnp.where(jnp.repeat(gmask, 4, axis=1), p1t, p2t)
    ca, cb = jnp.where(gmask, p1a, p2a), jnp.where(gmask, p1b, p2b)
    n = tab.shape[0]
    return (
        jnp.repeat(ct, 2, axis=0)[:n],
        jnp.repeat(ca, 2, axis=0)[:n],
        jnp.repeat(cb, 2, axis=0)[:n],
    )


def cone_crossover(key, net: Net, tab, wa, wb):
    """Cone crossover: a gate is inherited from the same parent as a consumer that reads it.

    Plain uniform crossover flips an independent coin per gate, which severs a gate from the
    consumers whose function was tuned to its output. Here parentage is decided at the OUTPUT layer
    (fair coins) and then propagated BACKWARDS along the fitter parent's wiring: each earlier-layer
    gate takes the parent of one of the gates that reads it (unread gates get a fresh coin). Whole
    input cones therefore tend to travel intact — the "building block" is a functional module, not an
    isolated gate. Wiring for the propagation is the first parent's (the child's wiring is mixed, so
    any single choice is an approximation; parent 1 is the tournament winner of the pair).
    """
    p1t, p2t = tab[0::2], tab[1::2]
    p1a, p2a, p1b, p2b = wa[0::2], wa[1::2], wb[0::2], wb[1::2]
    pairs = p1t.shape[0]
    sa, sb = jax.vmap(net.sources)(p1a, p1b)  # parent-1 resolved sources (pairs, n_gates)
    L = len(net.widths)
    key, ko = jax.random.split(key)
    masks = [None] * L
    masks[L - 1] = jax.random.bernoulli(ko, 0.5, (pairs, net.widths[-1]))
    for k in range(L - 2, -1, -1):
        lo, hi = net.offs[k + 1], net.offs[k + 2]  # the consumer layer above
        key, kf = jax.random.split(key)
        fresh = jax.random.bernoulli(kf, 0.5, (pairs, net.widths[k]))  # for unread gates

        def scatter(fresh_row, sa_row, sb_row, m_row):
            # consumers overwrite: last writer wins (arbitrary but consistent tie-break)
            out = fresh_row.at[sa_row].set(m_row)
            return out.at[sb_row].set(m_row)

        masks[k] = jax.vmap(scatter)(fresh, sa[:, lo:hi], sb[:, lo:hi], masks[k + 1])
    gmask = jnp.concatenate(masks, axis=1)  # (pairs, n_gates)
    ct = jnp.where(jnp.repeat(gmask, 4, axis=1), p1t, p2t)
    ca, cb = jnp.where(gmask, p1a, p2a), jnp.where(gmask, p1b, p2b)
    n = tab.shape[0]
    return (
        jnp.repeat(ct, 2, axis=0)[:n],
        jnp.repeat(ca, 2, axis=0)[:n],
        jnp.repeat(cb, 2, axis=0)[:n],
    )


def static_dead(net: Net, tab, wa, wb):
    """Data-independent dead mask for a whole population: (P, n_gates) bool.

    Same backward liveness as the deploy-time pruner, but batched over genomes: a gate is live iff a
    chain of statically-sensitive LUT ports connects it to an output bit. Static, so a dead gate
    cannot affect the circuit's output on ANY input — wiping it is behaviourally free.
    """
    code = net.codes(tab)  # (P, n_gates)
    sa, sb = jax.vmap(net.sources)(wa, wb)
    c0, c1, c2, c3 = (code & 1), ((code >> 1) & 1), ((code >> 2) & 1), ((code >> 3) & 1)
    sens_a = ((c0 ^ c2) | (c1 ^ c3)).astype(jnp.int8)  # responds to port a for some b
    sens_b = ((c0 ^ c1) | (c2 ^ c3)).astype(jnp.int8)

    P = tab.shape[0]
    live = [jnp.zeros((P, w), jnp.int8) for w in net.widths]
    live[-1] = jnp.ones((P, net.widths[-1]), jnp.int8)  # the popcount head reads every output bit
    for k in range(len(net.widths) - 2, -1, -1):
        lo, hi = net.offs[k + 1], net.offs[k + 2]  # consumer layer

        def scatter(a, b, ca, cb, cons):
            seg = jnp.zeros(net.widths[k], jnp.int8)
            seg = seg.at[a].max(ca * cons)
            return seg.at[b].max(cb * cons)

        live[k] = jax.vmap(scatter)(
            sa[:, lo:hi], sb[:, lo:hi], sens_a[:, lo:hi], sens_b[:, lo:hi], live[k + 1]
        )
    return jnp.concatenate(live, axis=1) == 0


def probe_logits(net: Net, tab, wa, wb, Xp):
    """Integer class logits on a fixed probe set: (P, classes, N). The basis of both descriptors."""
    fwd = jax.vmap(net._forward, in_axes=(0, 0, 0, None))
    return fwd(net.codes(tab), wa, wb, Xp).astype(jnp.int32)


def pop_softce(net: Net, tab, wa, wb, Xp, tgt):
    """Distillation fitness for a population: -softCE against target rows, per genome.

    The same soft-target CE the DID head minimises (z = popcounts / sqrt(group)), so under
    --distill every optimizer selects on the identical objective.
    """
    group = net.widths[-1] // net.classes
    z = probe_logits(net, tab, wa, wb, Xp).astype(jnp.float32)
    z = jnp.transpose(z, (0, 2, 1)) / group**0.5  # (P, N, C)
    lse = jax.scipy.special.logsumexp(z, axis=2)
    return -(lse - (z * tgt[None]).sum(2)).mean(1)


def descriptors(net: Net, logits):
    """MAP-Elites niche coordinates, both in [0,1] — BEHAVIOURAL, so they actually spread.

    Genome statistics (dead-gate fraction, wiring span) concentrate tightly around their
    random-init values, collapsing the archive into one cell — so the niches are defined on
    probe-set behaviour instead.

    d0 = prediction entropy over the probe set, normalised: 0 = the circuit collapses to one class,
         1 = it uses all ten. This is the axis a broken binary net actually varies along.
    d1 = mean output-bit activation rate (popcount / group size): how "on" the circuit's final layer
         sits. Independent of d0 and spreads widely at init.
    """
    group = net.widths[-1] // net.classes
    act = logits.mean((1, 2)) / group  # (P,) in [0,1]
    pred = logits.argmax(1)  # (P, N)
    hist = jnp.stack([(pred == c).mean(1) for c in range(net.classes)], 1)  # (P, C)
    ent = -(hist * jnp.log(hist + 1e-9)).sum(1) / jnp.log(net.classes)  # (P,) in [0,1]
    return jnp.stack([jnp.clip(ent, 0, 1), jnp.clip(act, 0, 1)], 1)  # (P, 2)


def behaviours(net: Net, logits):
    """NSLC behaviour descriptor: the class-prediction histogram on the probe set (10-dim).

    Two circuits with the same accuracy but different confusion structure are behaviourally distinct —
    exactly the difference novelty search exists to preserve.
    """
    pred = logits.argmax(1)
    return jnp.stack([(pred == c).mean(1) for c in range(net.classes)], 1)


# --------------------------------------------------------------------------------------
# DID — Discrete Influence Descent. Module-level and net-parameterised so the selftest can
# drive each piece directly; run_did jits partials of these.
# --------------------------------------------------------------------------------------
def gate_rows(code, a, b):
    """Evaluate one layer of 2-input LUT gates on packed rows: code (w,), a/b (w, N/8) uint8."""
    c = code[:, None]
    na, nb = ~a, ~b
    return (
        ((na & nb) * (c & 1))
        | ((na & b) * ((c >> 1) & 1))
        | ((a & nb) * ((c >> 2) & 1))
        | ((a & b) * ((c >> 3) & 1))
    )


def unpack_rows(packed):
    """(w, N/8) packed uint8 -> (N, w) float32 in {0,1} (inverse of `pack`, transposed)."""
    return jnp.unpackbits(packed, axis=1).T.astype(jnp.float32)


def tab_from_code(code):
    """Inverse of Net.codes for one genome: (n_gates,) codes -> (n_gates*4,) table bits."""
    g = jnp.stack([(code >> 3) & 1, (code >> 2) & 1, (code >> 1) & 1, code & 1], 1)
    return g.reshape(-1).astype(jnp.uint8)


def did_resolve(net: Net, wa, wb, lo, hi):
    """Wire values -> actual source indices for one layer slice (identity unless codebook)."""
    if not net.codebook:
        return wa[lo:hi], wb[lo:hi]
    ar = net.gate_ar[lo:hi]
    return net.cand_a[wa[lo:hi], ar], net.cand_b[wb[lo:hi], ar]


def did_forward_layers(net: Net, code, wa, wb, Xp):
    """Single-genome forward that keeps EVERY layer's packed activations (w_l, N/8)."""
    acts, prev = [], Xp
    for k in range(len(net.widths)):
        lo, hi = net.offs[k], net.offs[k + 1]
        sa, sb = did_resolve(net, wa, wb, lo, hi)
        prev = gate_rows(code[lo:hi], prev[sa], prev[sb])
        acts.append(prev)
    return acts


def did_head(net: Net, acts_last, y, targets=None):
    """Popcount-head CE loss + closed-form output influence lambda = dLoss/d(output bit).

    The head is fixed, so the output influence needs no autodiff: logits are the group popcounts
    scaled by 1/sqrt(group), giving per sample and bit
    lambda = (softmax - target)_class(bit) / (sqrt(group) * batch). The target row is the one-hot
    label or, under distillation, any distribution: the soft-target CE  lse - <t, z>  has the
    identical gradient form, so the whole influence/C machinery is target-agnostic.
    """
    group = net.widths[-1] // net.classes
    bits = jnp.unpackbits(acts_last, axis=1)  # (w_out, N)
    z = bits.reshape(net.classes, group, -1).sum(1).T.astype(jnp.float32) / group**0.5  # (N, C)
    n = z.shape[0]
    if targets is None:
        targets = jax.nn.one_hot(y, net.classes)
    lse = jax.scipy.special.logsumexp(z, axis=1)
    loss = (lse - (z * targets).sum(1)).mean()
    lam = (jax.nn.softmax(z, axis=1) - targets) / (group**0.5 * n)
    return loss, jnp.repeat(lam, group, axis=1)  # (N, w_out): bit j belongs to class j // group


def did_fitness(net: Net, acts_last, y):
    """GA-aligned batch objective from the packed last layer: (mean margin, accuracy).

    Margin = true-class popcount minus best-distractor popcount — the GA's fitness ingredients,
    so -(margin + w * acc) is a drop-in acceptance objective for the trial scan: at high accuracy
    CE decouples from the task metric, and accepting on the GA's own objective removes that
    mismatch while proposals still come from the CE surrogate.
    """
    bits = jnp.unpackbits(acts_last, axis=1)  # (w_out, N)
    group = net.widths[-1] // net.classes
    # float32 comparisons and boolean reductions only: popcounts are small exact integers, and
    # argmax(z) == y is spelled out with its first-max tie-break (no j < y ties or beats z_y,
    # nothing beats z_y) — identical values to an int argmax pipeline, but built from ops the
    # GPU compiler handles inside the rewire trial scan (argmax there segfaulted the backend)
    z = bits.reshape(net.classes, group, -1).sum(1).T.astype(jnp.float32)  # (N, C)
    oh = jax.nn.one_hot(y, net.classes)
    zt = (z * oh).sum(1)
    zo = (z - 1e9 * oh).max(1)
    margin = (zt - zo).mean()
    idx = jnp.arange(net.classes)[None, :]
    earlier_ge = (z >= zt[:, None]) & (idx < y[:, None])
    correct = ~earlier_ge.any(1) & (z <= zt[:, None]).all(1)
    acc = correct.astype(jnp.float32).mean()
    return margin, acc


def did_influence(net: Net, code, wa, wb, acts, Xp, lam_out):
    """Backward influence propagation with SIGNED gate sensitivities.

    lambda_u += lambda_j * (h(1,b) - h(0,b)) and symmetrically for input b — computed on the
    packed bits, accumulated per source with a segment_sum. Returns one (N, w_l) array per layer.

    The sensitivity must be the signed partial dh/da = h(1,b) - h(0,b) in {-1, 0, +1}, not the
    unsigned XOR dependency mask: lambda is a directional gradient, and a magnitude form silently
    discards the sign through inverting paths, corrupting every layer below the last
    (selftest 3b pins this against the multilinear-autodiff ground truth).
    """
    L = len(net.widths)
    lams = [None] * L
    lams[L - 1] = lam_out
    for li in range(L - 1, 0, -1):
        lo, hi = net.offs[li], net.offs[li + 1]
        sa, sb = did_resolve(net, wa, wb, lo, hi)
        prev = acts[li - 1]
        a, b = prev[sa], prev[sb]
        cl = code[lo:hi]
        ones, zeros = jnp.full_like(a, 255), jnp.zeros_like(a)
        sens_a = unpack_rows(gate_rows(cl, ones, b)) - unpack_rows(gate_rows(cl, zeros, b))
        sens_b = unpack_rows(gate_rows(cl, a, ones)) - unpack_rows(gate_rows(cl, a, zeros))
        ca, cb = lams[li] * sens_a, lams[li] * sens_b  # (N, w_l)
        pw = net.widths[li - 1]
        lams[li - 1] = (
            jax.ops.segment_sum(ca.T, sa, num_segments=pw)
            + jax.ops.segment_sum(cb.T, sb, num_segments=pw)
        ).T
    return lams


def did_head_gamma(net: Net, acts_last, y):
    """Diagonal head curvature per output bit: Gamma = p(1-p)_class(bit) / (group * n).

    Exact d2Loss/d(bit)^2 through the fixed popcount head (bit -> logit is linear with slope
    1/sqrt(group), the CE Hessian diagonal in logits is p(1-p)/n), like did_head's lambda but
    one derivative up. Always nonnegative. `y` is unused (the CE Hessian doesn't see the label)
    but kept for signature symmetry with did_head.
    """
    del y
    group = net.widths[-1] // net.classes
    bits = jnp.unpackbits(acts_last, axis=1)  # (w_out, N)
    z = bits.reshape(net.classes, group, -1).sum(1).T.astype(jnp.float32) / group**0.5  # (N, C)
    p = jax.nn.softmax(z, axis=1)
    gam = p * (1.0 - p) / (group * z.shape[0])
    return jnp.repeat(gam, group, axis=1)  # (N, w_out)


def did_curvature(net: Net, code, wa, wb, acts, Xp, gam_out):
    """Diagonal-GGN curvature backward: Gamma_u += Gamma_j * (dh/du)^2 per layer.

    The squared signed sensitivity (dh/du)^2 for dh/du in {-1, 0, +1} IS the unsigned XOR
    dependency mask — the magnitude form that corrupts lambda is exactly right for curvature,
    which is sign-free by construction. Each gate's multilinear extension is affine in each
    single input, so there is no local second-derivative term; mixed-parent and cross-child
    Hessian terms are dropped. This is a nonnegative damping oracle (second-order DID's
    C^(2) source), not exact Newton on the circuit.
    """
    L = len(net.widths)
    gams = [None] * L
    gams[L - 1] = gam_out
    for li in range(L - 1, 0, -1):
        lo, hi = net.offs[li], net.offs[li + 1]
        sa, sb = did_resolve(net, wa, wb, lo, hi)
        prev = acts[li - 1]
        a, b = prev[sa], prev[sb]
        cl = code[lo:hi]
        ones, zeros = jnp.full_like(a, 255), jnp.zeros_like(a)
        m_a = unpack_rows(gate_rows(cl, ones, b) ^ gate_rows(cl, zeros, b))  # (dh/da)^2 in {0,1}
        m_b = unpack_rows(gate_rows(cl, a, ones) ^ gate_rows(cl, a, zeros))
        ca, cb = gams[li] * m_a, gams[li] * m_b  # (N, w_l)
        pw = net.widths[li - 1]
        gams[li - 1] = (
            jax.ops.segment_sum(ca.T, sa, num_segments=pw)
            + jax.ops.segment_sum(cb.T, sb, num_segments=pw)
        ).T
    return gams


def code_bits(code):
    """(...,) uint8 codes -> (..., 4) float32 truth-table bits (pattern p = 2*a + b)."""
    return jnp.stack([(code >> p) & 1 for p in range(4)], -1).astype(jnp.float32)


def did_effective_c(c1, c2, cur_code):
    """Second-order effective coefficients: C~ = C1 + (1 - 2t)/2 * C2.

    Feeding C~ to did_best_response reproduces the curvature-damped flip rule exactly — a row
    flips iff sigma*C1 + C2/2 < 0 (sigma = 1-2t) — AND its returned delta equals the quadratic
    surrogate sum_p [C1_p dt_p + C2_p dt_p^2 / 2], since for a flipped row
    C~ * sigma = sigma*C1 + sigma^2 * C2/2 and sigma^2 = 1. C2 >= 0 makes it damping-only:
    marginal first-order flips are pruned, never added.
    """
    return c1 + 0.5 * (1.0 - 2.0 * code_bits(cur_code)) * c2


def did_layer_c(net: Net, li, lam_l, prev_acts, wa, wb):
    """C_{j,p} bins for every gate of one layer under its CURRENT parents.

    Direct masked sums, no cancellation forms — a pattern absent from the batch gives
    C_p == 0 exactly, not +/- rounding noise.
    """
    lo, hi = net.offs[li], net.offs[li + 1]
    sa, sb = did_resolve(net, wa, wb, lo, hi)
    Pf = unpack_rows(prev_acts)  # (N, prev_w)
    a, b = Pf[:, sa], Pf[:, sb]  # (N, w)
    na, nb = 1.0 - a, 1.0 - b
    return jnp.stack(
        [
            (lam_l * na * nb).sum(0),
            (lam_l * na * b).sum(0),
            (lam_l * a * nb).sum(0),
            (lam_l * a * b).sum(0),
        ],
        axis=1,
    )  # (w, 4)


def did_rewire_c(net: Net, li, lam_l, prev_acts, wa, wb):
    """Counterfactual C bins for every codebook candidate on each PORT of layer li's gates.

    Port-separable rewiring: Ca[g, k] holds gate g's four pattern bins with input a taken from
    structural candidate k while b keeps its current source — same linearisation as the table
    bins (lambda at g's output is untouched by rewiring g's inputs), so a candidate equal to the
    current wire reproduces did_layer_c exactly (selftest 15). Two einsums per port: the
    complementary patterns follow from the lambda column sums. Returns (Ca, Cb), each (w, K, 4).
    """
    lo, hi = net.offs[li], net.offs[li + 1]
    sa, sb = did_resolve(net, wa, wb, lo, hi)
    Pf = unpack_rows(prev_acts)  # (N, prev_w)
    a, b = Pf[:, sa], Pf[:, sb]  # (N, w)
    A = Pf[:, net.cand_a[:, lo:hi]]  # (N, K, w)
    B = Pf[:, net.cand_b[:, lo:hi]]
    lb, lnb = lam_l * b, lam_l * (1.0 - b)
    la, lna = lam_l * a, lam_l * (1.0 - a)
    Ca3 = jnp.einsum("nkw,nw->wk", A, lb)
    Ca2 = jnp.einsum("nkw,nw->wk", A, lnb)
    Ca1, Ca0 = lb.sum(0)[:, None] - Ca3, lnb.sum(0)[:, None] - Ca2
    Cb3 = jnp.einsum("nkw,nw->wk", B, la)
    Cb1 = jnp.einsum("nkw,nw->wk", B, lna)
    Cb2, Cb0 = la.sum(0)[:, None] - Cb3, lna.sum(0)[:, None] - Cb1
    Ca = jnp.stack([Ca0, Ca1, Ca2, Ca3], -1)
    Cb = jnp.stack([Cb0, Cb1, Cb2, Cb3], -1)
    return Ca, Cb


def did_rewire_c_joint(net: Net, li, lam_l, prev_acts):
    """Joint counterfactual C bins for every (u, v) candidate PAIR of layer li's gates.

    Cj[g, u, v] holds gate g's four pattern bins with input a from candidate u AND input b from
    candidate v — the two-port move the per-port bins cannot score when a good assignment needs
    both inputs changed together. One triple contraction gives the a=1,b=1 bin; the rest follow
    from the per-candidate and total lambda sums. Returns (w, K, K, 4).
    """
    lo, hi = net.offs[li], net.offs[li + 1]
    Pf = unpack_rows(prev_acts)  # (N, prev_w)
    A = Pf[:, net.cand_a[:, lo:hi]]  # (N, K, w)
    B = Pf[:, net.cand_b[:, lo:hi]]
    SA = jnp.einsum("nkw,nw->wk", A, lam_l)
    SB = jnp.einsum("nkw,nw->wk", B, lam_l)
    C3 = jnp.einsum("nuw,nvw->wuv", A * lam_l[:, None, :], B)
    C2 = SA[:, :, None] - C3
    C1 = SB[:, None, :] - C3
    C0 = lam_l.sum(0)[:, None, None] - SA[:, :, None] - SB[:, None, :] + C3
    return jnp.stack([C0, C1, C2, C3], -1)


def did_rewire_trial_scan(
    net: Net,
    code,
    sa0,
    sb0,
    Xp,
    y,
    g,
    t,
    g2,
    t2,
    port,
    snew_a,
    snew_b,
    valid,
    loss0,
    fit_w=None,
    targets=None,
):
    """Trial scan over a mixed pool of table, motif, and rewire proposals, on RESOLVED sources.

    Each step writes tables at (g, g2) — a single-gate proposal is encoded with both slots equal,
    a parent-child motif carries its two gates — and the rewired port(s)' resolved source index
    from (snew_a, snew_b): port 0 sets a, port 1 sets b, port 2 sets both (a joint pair), port -1
    is a pure table proposal (sources ignored). Carrying resolved sources instead of wire choices
    keeps the codebook lookup out of the scan body — the caller resolves each proposal's
    candidate(s) up front and reconstructs (wa, wb) from the returned accept mask. Acceptance is
    strict on the carried objective — CE, or the GA objective with fit_w. Returns the final
    codes, sources, loss, counts, and the accept mask.
    """

    def full_loss(cd, sa, sb, X, yy):
        prev = X
        for k in range(len(net.widths)):
            lo, hi = net.offs[k], net.offs[k + 1]
            prev = gate_rows(cd[lo:hi], prev[sa[lo:hi]], prev[sb[lo:hi]])
        if fit_w is None:
            return did_head(net, prev, yy, targets)[0]
        margin, acc = did_fitness(net, prev, yy)
        return -(margin + fit_w * acc)

    def step(carry, x):
        gg, tt, gg2, tt2, pp, ssa, ssb, ok = x

        def trial(op):
            cd, sa, sb, cur, n_tr, n_ac = op
            nc = cd.at[gg].set(tt).at[gg2].set(tt2)
            na = sa.at[gg].set(jnp.where((pp == 0) | (pp == 2), ssa, sa[gg]))
            nb = sb.at[gg].set(jnp.where((pp == 1) | (pp == 2), ssb, sb[gg]))
            tl = full_loss(nc, na, nb, Xp, y)
            acc = tl < cur
            return (
                jnp.where(acc, nc, cd),
                jnp.where(acc, na, sa),
                jnp.where(acc, nb, sb),
                jnp.where(acc, tl, cur),
                n_tr + 1,
                n_ac + acc.astype(jnp.int32),
            ), acc

        return jax.lax.cond(ok, trial, lambda op: (op, jnp.bool_(False)), carry)

    init = (code, sa0, sb0, jnp.float32(loss0), jnp.int32(0), jnp.int32(0))
    (code, sa0, sb0, loss, n_tr, n_ac), accepted = jax.lax.scan(
        step, init, (g, t, g2, t2, port, snew_a, snew_b, valid)
    )
    return code, sa0, sb0, loss, n_tr, n_ac, accepted


def did_rewire_snew(net: Net, g, p, k):
    """Resolved new-source pair per proposal: the rewired port(s) get cand[.., g], others 0.

    k holds a single candidate index for ports 0/1 and an encoded pair u*K + v for port 2.
    """
    K = net.codebook
    ca, cb = np.asarray(net.cand_a), np.asarray(net.cand_b)
    u = np.where(p == 2, k // K, k)
    v = np.where(p == 2, k % K, k)
    sna = np.where((p == 0) | (p == 2), ca[u, g], 0).astype(np.int32)
    snb = np.where((p == 1) | (p == 2), cb[v, g], 0).astype(np.int32)
    return sna, snb


def did_rewire_apply(net: Net, wa, wb, g, p, k, accepted):
    """Replay accepted rewires into the wire-choice genome, in scan order (host-side)."""
    K = net.codebook
    wa_np, wb_np = np.asarray(wa).copy(), np.asarray(wb).copy()
    for i in np.flatnonzero(np.asarray(accepted)):
        if p[i] == 0:
            wa_np[g[i]] = k[i]
        elif p[i] == 1:
            wb_np[g[i]] = k[i]
        elif p[i] == 2:
            wa_np[g[i]] = k[i] // K
            wb_np[g[i]] = k[i] % K
    return jnp.asarray(wa_np), jnp.asarray(wb_np)


def did_rewire_pool(net: Net, code, wa, wb, pcode, delta, cab, c_layers, cjs=None, motifs=None):
    """Mixed proposal pool: table singletons (port -1), per-port rewires, joint pairs (port 2),
    and optional parent-child motifs (port -1, two gates).

    cab = per-layer (Ca, Cb) counterfactual bins, c_layers = per-layer current bins, cjs =
    optional per-layer joint (u, v) bins, motifs = optional flattened (g1, t1, g2, t2, delta)
    parent-child bundles. A rewire's surrogate delta is its candidate best-response delta plus
    the base correction sum_p t_p (C_k - C_cur), so all proposal types are measured against the
    same linearisation; the candidate equal to the current wire is masked to 0 (it IS the
    singleton), and joint pairs keeping either port current are masked to 0 (they duplicate the
    per-port entries). Returns np arrays (g, t, g2, t2, port, k, delta) — single-gate entries
    duplicate (g, t) into (g2, t2); a joint pair is encoded k = u*K + v.
    """
    K = net.codebook
    gi = jnp.arange(net.n_gates, dtype=jnp.int32)
    g_a = [gi]
    t_a = [pcode]
    p_a = [jnp.full(net.n_gates, -1, jnp.int32)]
    k_a = [jnp.zeros(net.n_gates, jnp.int32)]
    d_a = [delta]
    for li in range(len(net.widths)):
        lo, hi = net.offs[li], net.offs[li + 1]
        Ca, Cb = cab[li]
        code_l = code[lo:hi]
        gb = code_bits(code_l)
        w_ar = jnp.arange(hi - lo)
        for port, C, curw in ((0, Ca, wa[lo:hi]), (1, Cb, wb[lo:hi])):
            pk, dk = did_best_response(C, code_l[:, None])  # (w, K)
            base = ((C - c_layers[li][:, None, :]) * gb[:, None, :]).sum(-1)
            move = (dk + base).at[w_ar, curw].set(0.0)
            g_a.append(jnp.repeat(gi[lo:hi], K))
            t_a.append(pk.reshape(-1))
            p_a.append(jnp.full((hi - lo) * K, port, jnp.int32))
            k_a.append(jnp.tile(jnp.arange(K, dtype=jnp.int32), hi - lo))
            d_a.append(move.reshape(-1))
        if cjs is not None:
            Cj = cjs[li]  # (w, K, K, 4)
            pkj, dkj = did_best_response(Cj, code_l[:, None, None])  # (w, K, K)
            base_j = ((Cj - c_layers[li][:, None, None, :]) * gb[:, None, None, :]).sum(-1)
            move_j = dkj + base_j
            move_j = move_j.at[w_ar, wa[lo:hi], :].set(0.0).at[w_ar, :, wb[lo:hi]].set(0.0)
            g_a.append(jnp.repeat(gi[lo:hi], K * K))
            t_a.append(pkj.reshape(-1))
            p_a.append(jnp.full((hi - lo) * K * K, 2, jnp.int32))
            k_a.append(jnp.tile(jnp.arange(K * K, dtype=jnp.int32), hi - lo))
            d_a.append(move_j.reshape(-1))
    g2_a, t2_a = [jnp.concatenate(g_a)], [jnp.concatenate(t_a)]  # single-gate: both slots equal
    if motifs is not None:
        mg1, mt1, mg2, mt2, mdl = motifs
        g_a, t_a = [g2_a[0], mg1], [t2_a[0], mt1]
        g2_a.append(mg2)
        t2_a.append(mt2)
        p_a.append(jnp.full(mg1.shape[0], -1, jnp.int32))
        k_a.append(jnp.zeros(mg1.shape[0], jnp.int32))
        d_a.append(mdl)
    return tuple(np.asarray(jnp.concatenate(v)) for v in (g_a, t_a, g2_a, t2_a, p_a, k_a, d_a))


def did_dedup_order(order, g1, g2, cap):
    """Keep only the best-ranked proposal per written gate, up to cap.

    The ranked top concentrates many proposals on the same gates (each gate fields a singleton,
    2K rewires, K^2 pairs, motifs); once the best one accepts, the rest are stale and burn their
    trials. Walking the full ranking and claiming each written gate keeps the trial list on cap
    DISTINCT gates instead.
    """
    seen, keep = set(), []
    for i in order:
        a, b = int(g1[i]), int(g2[i])
        if a in seen or b in seen:
            continue
        keep.append(i)
        seen.add(a)
        seen.add(b)
        if len(keep) >= cap:
            break
    return np.asarray(keep, dtype=order.dtype)


def did_best_response(C, cur_code):
    """Closed-form best table under bins C, changing ONLY bits a nonzero coefficient drives.

    t_p = 1 iff C_p < 0, t_p = 0 iff C_p > 0, and C_p == 0 keeps the current bit — proposals are
    C-driven, never zero-evidence flips. Returns the proposed codes and the surrogate delta
    sum_p C_p (t_p - g_p), strictly negative iff anything changes. Broadcasts: C (..., 4) against
    cur_code (...) — e.g. (w, 16, 4) bins against (w, 1) codes for per-candidate best responses.
    """
    g = code_bits(cur_code)
    t = jnp.where(C < 0, 1.0, jnp.where(C > 0, 0.0, g))
    delta = ((t - g) * C).sum(-1)
    tb = t.astype(jnp.uint8)
    prop = tb[..., 0] | (tb[..., 1] << 1) | (tb[..., 2] << 2) | (tb[..., 3] << 3)
    return prop.astype(jnp.uint8), delta


def did_motif_props(net: Net, li, lam_l, pin, pacts, code, wa, wb, c_parent):
    """Counterfactual parent-child proposals for every edge into child layer `li` (>= 1).

    For an edge j -> k (parent gate j in layer li-1 feeding one input slot of child k): each of
    the 16 candidate parent tables h_j is rolled out counterfactually through the parent's cached
    inputs (z = h_j(a_j, b_j), never a network forward), the child's C bins are re-binned in that
    world with the child's OTHER input r taken from the current activations, and the child's best
    response h_k*(.|h_j) is closed-form. The bundle score is
        delta(h_j) = [S_j(h_j) - S_j(g_j)] + [S_{k|h_j}(h_k*) - S_{k|h_j}(g_k)]
    (both child alternatives under the SAME candidate parent, so the rerouting is not credited
    twice; the no-op normalises to 0). Each edge proposes its argmin h_j as ONE two-gate change.
    h_j == g_j is masked out of the argmin: with the parent unchanged the motif degenerates to
    the child's singleton, which is already in the pool — so every motif genuinely changes the
    parent. Edges whose child reads the same parent on both slots are also skipped (the
    counterfactual would have to flow through both inputs).

    Args:
        lam_l: (N, w_child) child-layer influence.
        pin: packed input rows TO the parent layer (Xp for li == 1, else acts[li-2]).
        pacts: (w_parent, N/8) the parent layer's current activations.
        c_parent: (w_parent, 4) the parent layer's C bins (for S_j).

    Returns:
        (g1, c1, g2, c2, delta): parent global index + proposed table, child global index +
        proposed table, and the bundle's surrogate delta — both slots concatenated.
    """
    lp = li - 1
    lo_c, hi_c = net.offs[li], net.offs[li + 1]
    lo_p, hi_p = net.offs[lp], net.offs[lp + 1]
    w_c, w_p = hi_c - lo_c, hi_p - lo_p
    hbits = code_bits(jnp.arange(16, dtype=jnp.uint8))  # (16, 4)
    gp = code_bits(code[lo_p:hi_p])  # (w_p, 4)
    dP = ((hbits[None] - gp[:, None]) * c_parent[:, None, :]).sum(-1)  # (w_p, 16)
    sa_p, sb_p = did_resolve(net, wa, wb, lo_p, hi_p)
    Z = jnp.stack(
        [gate_rows(jnp.full((w_p,), h, jnp.uint8), pin[sa_p], pin[sb_p]) for h in range(16)]
    )  # (16, w_p, N/8) counterfactual parent outputs
    csa, csb = did_resolve(net, wa, wb, lo_c, hi_c)
    gc = code[lo_c:hi_c]
    afr = unpack_rows(pacts[csa])  # (N, w_c) child input a, current world
    bfr = unpack_rows(pacts[csb])
    ar = jnp.arange(w_c)
    out = []
    for slot_a, cs, rf in ((True, csa, bfr), (False, csb, afr)):
        zf = jax.vmap(unpack_rows)(Z[:, cs])  # (16, N, w_c)
        nzf = 1.0 - zf
        lam_r, lam_nr = lam_l * rf, lam_l * (1.0 - rf)
        # direct masked sums (no cancellation): an absent counterfactual pattern's C is exactly 0
        Czr = jnp.einsum("hnw,nw->wh", zf, lam_r)  # z = 1, r = 1   -> (w_c, 16)
        Czn = jnp.einsum("hnw,nw->wh", zf, lam_nr)  # z = 1, r = 0
        Cnr = jnp.einsum("hnw,nw->wh", nzf, lam_r)  # z = 0, r = 1
        Cnn = jnp.einsum("hnw,nw->wh", nzf, lam_nr)  # z = 0, r = 0
        # pattern index p = 2*a + b; z sits on slot a or slot b
        bins = jnp.stack([Cnn, Cnr, Czn, Czr] if slot_a else [Cnn, Czn, Cnr, Czr], -1)
        ck, dC = did_best_response(bins, gc[:, None])  # (w_c, 16) each
        d = jnp.where((csa == csb)[:, None], jnp.inf, dP[cs] + dC)  # (w_c, 16)
        h_all = jnp.arange(16, dtype=jnp.uint8)
        d = jnp.where(h_all[None] == code[lo_p:hi_p][cs][:, None], jnp.inf, d)
        h_star = jnp.argmin(d, axis=1)
        out.append(
            (
                lo_p + cs,
                h_star.astype(jnp.uint8),
                lo_c + ar,
                ck[ar, h_star],
                d[ar, h_star],
            )
        )
    return tuple(jnp.concatenate([o[i] for o in out]) for i in range(5))


def did_trial_scan(
    net: Net,
    code,
    wa,
    wb,
    Xp,
    y,
    g1,
    c1,
    g2,
    c2,
    valid,
    loss0,
    unif=None,
    temp=0.0,
    Xp2=None,
    y2=None,
    loss2=None,
    fit_w=None,
    targets=None,
):
    """Acceptance over a priority-ordered proposal list, ONE proposal at a time.

    Each step applies its proposal alone (two .at[].set writes — a singleton is encoded with both
    slots equal, so the second write is a no-op), runs a full forward on the batch, and keeps the
    change only if the real loss STRICTLY falls — or, when `unif`/`temp` are given, by the
    Metropolis rule: a worse trial is also accepted with probability exp(-delta / temp), so the
    carried loss is monotone ONLY in the exact (default) mode. With a CONFIRM batch (Xp2/y2 and
    its anchor loss2), a trial must additionally strictly improve the loss on that second,
    independent batch — the guard against single-batch acceptance churn; each trial then costs
    two forwards. With `fit_w`, the accepted quantity is not CE but the GA's own objective,
    carried as -(margin + fit_w * accuracy) — proposals still come from the CE surrogate, but
    acceptance aligns with the task metric (loss0/loss2 anchors must then be that objective).
    A proposal made stale by an earlier accept in the same sweep simply fails its trial.
    Returns the final genome codes, loss, (trials, accepts), and the per-step accept mask
    (for host-side per-proposal-type diagnostics).
    """
    if unif is None:
        unif = jnp.full(valid.shape, 2.0)  # u >= 1 can never pass the Metropolis clause
    temp = jnp.maximum(jnp.float32(temp), 1e-12)  # exact mode: exp(-delta/1e-12) underflows to 0
    confirm = Xp2 is not None
    if not confirm:
        loss2 = jnp.float32(0.0)

    def full_loss(cd, X, yy, t):
        prev = X
        for k in range(len(net.widths)):
            lo, hi = net.offs[k], net.offs[k + 1]
            sa, sb = did_resolve(net, wa, wb, lo, hi)
            prev = gate_rows(cd[lo:hi], prev[sa], prev[sb])
        if fit_w is None:
            return did_head(net, prev, yy, t)[0]
        margin, acc = did_fitness(net, prev, yy)
        return -(margin + fit_w * acc)

    def step(carry, x):
        p1, t1, p2, t2, ok, u = x

        def trial(op):
            cd, cur_loss, cur_loss2, n_tr, n_ac = op
            new = cd.at[p1].set(t1).at[p2].set(t2)
            tl = full_loss(new, Xp, y, targets)
            acc = (tl < cur_loss) | (u < jnp.exp((cur_loss - tl) / temp))
            tl2 = cur_loss2
            if confirm:
                tl2 = full_loss(new, Xp2, y2, None)  # confirm batch stays label-CE
                acc = acc & (tl2 < cur_loss2)
            return (
                jnp.where(acc, new, cd),
                jnp.where(acc, tl, cur_loss),
                jnp.where(acc, tl2, cur_loss2),
                n_tr + 1,
                n_ac + acc.astype(jnp.int32),
            ), acc

        return jax.lax.cond(ok, trial, lambda op: (op, jnp.bool_(False)), carry)

    init = (code, loss0, loss2, jnp.int32(0), jnp.int32(0))
    (code, loss, _, n_tr, n_ac), accepted = jax.lax.scan(step, init, (g1, c1, g2, c2, valid, unif))
    return code, loss, n_tr, n_ac, accepted


# --------------------------------------------------------------------------------------
# The algorithms
# --------------------------------------------------------------------------------------
class Runner:
    def __init__(self, net: Net, Xtr, ytr, Xte, yte, cfg: Config):
        self.net, self.cfg = net, cfg
        self.Xtr, self.ytr = Xtr, ytr
        self.Xte_p, self.yte = pack(Xte), yte
        self.key = jax.random.PRNGKey(cfg.seed)
        self.best = (None, 0.0)  # (genome, test acc)
        self.evals = 0
        pk = jax.random.PRNGKey(cfg.seed + 99)
        idx = jax.random.randint(pk, (cfg.probe,), 0, Xtr.shape[0])
        self.probe_p = pack(Xtr[idx])
        self.desc = jax.jit(
            lambda t, a, b: descriptors(net, probe_logits(net, t, a, b, self.probe_p))
        )
        self.behav = jax.jit(
            lambda t, a, b: behaviours(net, probe_logits(net, t, a, b, self.probe_p))
        )
        self.tgt_all = None
        if cfg.distill:
            td = np.load(cfg.distill)
            assert (np.asarray(ytr) == td["y_train"]).all(), (
                "teacher logits misaligned with the training set"
            )
            q = jax.nn.softmax(jnp.asarray(td["train_logits"], jnp.float32) / cfg.distill_temp, 1)
            oh = jax.nn.one_hot(ytr, net.classes)
            self.tgt_all = (1.0 - cfg.distill_alpha) * oh + cfg.distill_alpha * q
        self.softce_fit = jax.jit(functools.partial(pop_softce, net))

    def split(self, n):
        self.key, *ks = jax.random.split(self.key, n + 1)
        return ks

    def fitness(self, tab, wa, wb, kb, m: int = 1):
        """One evaluation of a batch of genomes on a fresh minibatch (the shared fitness).

        With m > 1 the estimate averages m INDEPENDENT fresh base-size batches — samples are iid
        draws with replacement, so this is statistically identical to one m-times-larger batch,
        but reuses the same compiled kernel at the same peak memory. Charged m evaluations per
        genome. Under --distill the shared fitness IS the distillation objective (-softCE
        against the teacher's target rows) — selection on the same loss the DID head descends."""
        codes = self.net.codes(tab)
        acc_f = 0.0
        for k in jax.random.split(kb, m):
            idx = jax.random.randint(k, (self.cfg.batch,), 0, self.Xtr.shape[0])
            if self.tgt_all is not None:
                acc_f = acc_f + self.softce_fit(tab, wa, wb, pack(self.Xtr[idx]), self.tgt_all[idx])
            else:
                margin, acc = self.net.eval_pop(codes, wa, wb, pack(self.Xtr[idx]), self.ytr[idx])
                acc_f = acc_f + margin + self.cfg.acc_weight * acc
        self.evals += tab.shape[0] * m
        return acc_f / m

    def batch_at(self, gen):
        """Stepwise-doubling effective-batch schedule: multiplier 1 for the first half of the
        BUDGET, then equal budget-share stages doubling to batch_end/batch fresh base-size
        batches averaged per fitness call."""
        c = self.cfg
        if not c.batch_end or c.batch_end <= c.batch:
            return 1
        d = max(1, round(float(np.log2(c.batch_end / c.batch))))
        f = self.progress(gen)
        if f < 0.5:
            return 1
        stage = min(d, 1 + int((f - 0.5) / (0.5 / d)))
        return 2**stage

    def test_best(self, tab, wa, wb):
        _, acc = self.net.eval_pop(
            self.net.codes(tab[None]), wa[None], wb[None], self.Xte_p, self.yte
        )
        v = float(acc[0])
        if v > self.best[1]:
            self.best = ((tab, wa, wb), v)
        return v

    def progress(self, gen):
        """Anneal progress in [0, 1]. With batch_end/elite_reval the per-gen eval charge varies,
        so the run ends when the BUDGET does, not at gens — progress follows spent budget there
        (identical to gen/gens under uniform charging)."""
        c = self.cfg
        if c.batch_end or c.elite_reval:
            return min(1.0, self.evals / (c.pop * c.gens))
        return gen / max(c.gens - 1, 1)

    def rates(self, gen):
        f = self.progress(gen)
        c = self.cfg
        return c.mut * (c.mut_end / c.mut) ** f, c.wire_mut * (c.wire_mut_end / c.wire_mut) ** f

    def log(self, gen, fit, extra=""):
        if gen % self.cfg.log_every == 0 or gen == self.cfg.gens - 1:
            v = self.test_best(*self.cur_best())
            print(
                f"gen {gen:5d}  fit best {float(jnp.max(fit)):8.2f}  mean {float(jnp.mean(fit)):8.2f}"
                f"  TEST {v:.4f}  (best {self.best[1]:.4f}) {extra}",
                flush=True,
            )

    # ---- ga: tournament + crossover + elitism (the incumbent) -------------------------
    def run_ga(self):
        """The incumbent GA. With bias_beta > 0, crossover is IMPORTANCE-BIASED: every bias_every
        generations the validated sensitivity backward pass scores each gate on the current best
        genome, and important gates are inherited from the FITTER parent of each pair with
        probability 0.5 + bias_beta * normalised_importance (unimportant genes keep mixing 50/50,
        preserving diversity)."""
        c, net = self.cfg, self.net
        (ki,) = self.split(1)
        tab, wa, wb = net.init_pop(ki, c.pop)
        if c.init_genome:
            # warm start: half the population = mutated copies of the genome (slot 0 exact),
            # the other half keeps its random init as a diversity hedge
            gt, gwa, gwb = load_genome(c.init_genome)
            (ks,) = self.split(1)
            half = max(1, c.pop // 2)
            st, sa2, sb2 = mutate(
                ks,
                net,
                jnp.tile(gt[None], (half, 1)),
                jnp.tile(gwa[None], (half, 1)),
                jnp.tile(gwb[None], (half, 1)),
                0.02,
                0.01,
            )
            tab = tab.at[:half].set(st).at[0].set(gt)
            wa = wa.at[:half].set(sa2).at[0].set(gwa)
            wb = wb.at[:half].set(sb2).at[0].set(gwb)
        ar = jnp.arange(c.pop)
        imp = None  # per-gate importance in [0,1] (refreshed every bias_every gens)
        jits = self._did_jits() if (c.memetic_every or c.mut_influence > 0) else None
        tbl16 = code_bits(jnp.arange(16, dtype=jnp.uint8))  # (16, 4)
        mut_logits = None  # (n_gates, 16) influence-sampled mutation kernel
        hist = {}  # elite_hist: genome-hash -> (sum, sumsq, n) of centered fitness
        sd_prior = 0.33  # per-genome single-batch sd prior until a genome has its own history
        target = c.pop * c.gens
        for gen in range(c.gens):
            if self.evals >= target:
                break  # burst trials are charged to the same budget
            mut, wmut = self.rates(gen)
            kb, kt, kc, km = self.split(4)
            if c.sterilize_every and gen and gen % c.sterilize_every == 0:
                # wiping dead gates is behaviourally free (static liveness), so any accuracy
                # change isolates the value of the content they were holding
                dead = static_dead(net, tab, wa, wb)
                (ks,) = self.split(1)
                fresh = (
                    jax.random.bernoulli(ks, 0.5, tab.shape).astype(tab.dtype)
                    if c.sterilize_mode == "rand"
                    else jnp.zeros_like(tab)
                )
                tab = jnp.where(jnp.repeat(dead, 4, axis=1), fresh, tab)
            m_bs = self.batch_at(gen)
            fit = self.fitness(tab, wa, wb, kb, m_bs)
            order = jnp.argsort(-fit)
            tab, wa, wb, fit = tab[order], wa[order], wb[order], fit[order]
            if c.elite_reval:
                # refine the top-16 on extra independent batches and re-rank THEM by the average
                # (elite retention + false-positive guard); tournaments keep the single-batch fit
                # for everyone, so no cross-batch common-mode offset enters the selection pressure
                R = min(16, c.pop)
                ref = fit[:R]
                for kr in self.split(c.elite_reval):
                    ref = ref + self.fitness(tab[:R], wa[:R], wb[:R], kr, m_bs)
                ro = jnp.argsort(-(ref / (1 + c.elite_reval)))
                tab = tab.at[:R].set(tab[:R][ro])
                wa = wa.at[:R].set(wa[:R][ro])
                wb = wb.at[:R].set(wb[:R][ro])
                fit = fit.at[:R].set(fit[:R][ro])
            if c.elite_hist > 0:
                # LCB elitism: every generation IS an independent batch, so surviving elites
                # accumulate free re-evaluations. Rank the top-16 by history mean of centered
                # fitness (median removed = batch common mode) minus z * sd/sqrt(n): a one-batch
                # challenger must clear a significance bar to displace a well-measured incumbent.
                R = min(16, c.pop)
                med = float(jnp.median(fit))
                fr = np.asarray(fit[:R], np.float64)
                lcb = np.empty(R)
                for i in range(R):
                    kh = hash(
                        (
                            np.asarray(tab[i]).tobytes(),
                            np.asarray(wa[i]).tobytes(),
                            np.asarray(wb[i]).tobytes(),
                        )
                    )
                    s, ss, n = hist.get(kh, (0.0, 0.0, 0))
                    v = fr[i] - med
                    s, ss, n = s + v, ss + v * v, n + 1
                    hist[kh] = (s, ss, n)
                    m = s / n
                    if n >= 3:
                        sd = max((ss / n - m * m), 0.0) ** 0.5
                        sd_prior = 0.98 * sd_prior + 0.02 * sd
                    else:
                        sd = sd_prior
                    lcb[i] = m - c.elite_hist * sd / n**0.5
                ro = jnp.asarray(np.argsort(-lcb))
                tab = tab.at[:R].set(tab[:R][ro])
                wa = wa.at[:R].set(wa[:R][ro])
                wb = wb.at[:R].set(wb[:R][ro])
                fit = fit.at[:R].set(fit[:R][ro])
                if len(hist) > 200_000:  # long converged runs: drop the cold entries
                    hist = dict(sorted(hist.items(), key=lambda e: e[1][2])[100_000:])
            self.cur = (tab[0], wa[0], wb[0])
            if c.memetic_every and gen and gen % c.memetic_every == 0:
                (kd,) = self.split(1)
                code0, nwa, nwb = self._did_burst(jits, net.codes(tab[:1])[0], wa[0], wb[0], kd)
                tab = tab.at[0].set(tab_from_code(code0))
                wa = wa.at[0].set(nwa)
                wb = wb.at[0].set(nwb)
                self.cur = (tab[0], wa[0], wb[0])
            if c.mut_influence > 0 and gen % c.bias_every == 0:
                (kd,) = self.split(1)
                c_all = self._did_pass(jits, net.codes(tab[:1])[0], wa[0], wb[0], kd)[3]
                s = jnp.abs(c_all).max(1, keepdims=True) + 1e-12  # per-gate scale (dead -> uniform)
                mut_logits = -(c_all / (c.mut_influence * s)) @ tbl16.T  # (n_gates, 16)
            if c.bias_beta > 0 and gen % c.bias_every == 0:
                from gate_importance_mnist import sensitivity  # lazy: avoids a circular import

                (kp,) = self.split(1)
                pi = jax.random.randint(kp, (c.probe,), 0, self.Xtr.shape[0])
                s = sensitivity(net, tab[0], wa[0], wb[0], self.Xtr[pi])
                imp = s / (s.max() + 1e-9)  # (n_gates,) in [0,1]
            self.log(
                gen, fit, extra=f"imp>0 {float((imp > 0).mean()):.2f}" if imp is not None else ""
            )
            cand = jax.random.randint(kt, (c.pop, c.tsize), 0, c.pop)
            win = cand[ar, fit[cand].argmax(1)]
            pt, pa, pb = tab[win], wa[win], wb[win]
            if c.use_crossover:
                if c.cone:
                    ct, ca, cb = cone_crossover(kc, net, pt, pa, pb)
                elif imp is None:
                    ct, ca, cb = crossover(kc, net, pt, pa, pb)
                else:
                    # P(child takes gate g from the FITTER parent) = 0.5 + beta * imp[g]
                    f1, f2 = fit[win][0::2], fit[win][1::2]  # each pair's parent fitnesses
                    p_take1 = jnp.where(
                        (f1 >= f2)[:, None], 0.5 + c.bias_beta * imp, 0.5 - c.bias_beta * imp
                    )  # (pairs, n_gates)
                    gmask = jax.random.uniform(kc, p_take1.shape) < p_take1
                    ct = jnp.where(jnp.repeat(gmask, 4, axis=1), pt[0::2], pt[1::2])
                    ca = jnp.where(gmask, pa[0::2], pa[1::2])
                    cb = jnp.where(gmask, pb[0::2], pb[1::2])
                    ct = jnp.repeat(ct, 2, axis=0)[: c.pop]
                    ca = jnp.repeat(ca, 2, axis=0)[: c.pop]
                    cb = jnp.repeat(cb, 2, axis=0)[: c.pop]
            else:
                ct, ca, cb = pt, pa, pb
            ct, ca, cb = mutate(km, net, ct, ca, cb, 0.0 if mut_logits is not None else mut, wmut)
            if mut_logits is not None:
                # influence-sampled table mutation: same expected number of mutated gates as the
                # blind bit-flip operator (4 bits at rate mut), but the new table is drawn from
                # the current-best genome's C kernel instead of flipping random bits
                k1, k2 = self.split(2)
                codes = net.codes(ct)
                gm = jax.random.bernoulli(k1, 4.0 * mut, codes.shape)
                samp = jax.random.categorical(
                    k2, jnp.broadcast_to(mut_logits, (c.pop,) + mut_logits.shape), axis=-1
                ).astype(jnp.uint8)
                ct = jax.vmap(tab_from_code)(jnp.where(gm, samp, codes))
            tab = ct.at[: c.elite].set(pt[: c.elite])
            wa = ca.at[: c.elite].set(pa[: c.elite])
            wb = cb.at[: c.elite].set(pb[: c.elite])

    # ---- aging: evict the OLDEST, not the worst ---------------------------------------
    def run_aging(self):
        """Population of `pop`; each iteration births a cohort of pop/4 and overwrites the oldest.

        The cohort must be SMALLER than the population, otherwise every member is replaced each
        iteration and nothing ever ages — it collapses into plain generational replacement. A circular
        write pointer gives exact FIFO eviction: the slots we overwrite are always the oldest ones.
        Iterations are scaled up so the evaluation budget matches the other algorithms.
        """
        c, net = self.cfg, self.net
        cohort = max(1, c.pop // 4)
        iters = c.gens * c.pop // cohort  # same total evaluations as the GA
        (ki,) = self.split(1)
        tab, wa, wb = net.init_pop(ki, c.pop)
        (kb0,) = self.split(1)
        fit = self.fitness(
            tab, wa, wb, kb0
        )  # fitness recorded at birth and kept (no re-evaluation)
        ar = jnp.arange(cohort)
        ptr = 0
        for it in range(iters):
            gen = it * cohort // c.pop  # progress on the shared anneal schedule
            mut, wmut = self.rates(gen)
            kb, kt, km = self.split(3)
            cand = jax.random.randint(
                kt, (cohort, c.tsize), 0, c.pop
            )  # tournament among the living
            win = cand[ar, fit[cand].argmax(1)]
            ct, ca, cb = mutate(km, net, tab[win], wa[win], wb[win], mut, wmut)  # mutation-only
            cfit = self.fitness(ct, ca, cb, kb)
            slots = (ptr + jnp.arange(cohort)) % c.pop  # the oldest `cohort` slots
            tab, wa, wb = tab.at[slots].set(ct), wa.at[slots].set(ca), wb.at[slots].set(cb)
            fit = fit.at[slots].set(cfit)
            ptr = (ptr + cohort) % c.pop
            i = int(jnp.argmax(fit))
            self.cur = (tab[i], wa[i], wb[i])
            if it % max(1, (iters // (c.gens // c.log_every))) == 0 or it == iters - 1:
                v = self.test_best(*self.cur_best())
                print(
                    f"iter {it:6d}  fit best {float(jnp.max(fit)):8.2f}  mean {float(jnp.mean(fit)):8.2f}"
                    f"  TEST {v:.4f}  (best {self.best[1]:.4f})",
                    flush=True,
                )

    # ---- map-elites: an archive of niches, not a population ---------------------------
    def run_mapelites(self):
        c, net = self.cfg, self.net
        G = c.grid
        a_tab = np.zeros((G * G, net.n_bits), np.uint8)
        a_wa = np.zeros((G * G, net.n_gates), np.int32)
        a_wb = np.zeros((G * G, net.n_gates), np.int32)
        a_fit = np.full(G * G, -np.inf, np.float32)

        def insert(tab, wa, wb, fit, d):
            """Vectorised elite insertion: per niche keep the single best candidate, if it beats the
            incumbent. (A per-candidate Python loop here would run pop*gens = 10M times.)"""
            cell = np.clip((np.asarray(d) * G).astype(int), 0, G - 1)
            cid = cell[:, 0] * G + cell[:, 1]
            f = np.asarray(fit)
            order = np.argsort(-f)  # best-first, so unique() keeps the best per niche
            uniq, first = np.unique(cid[order], return_index=True)
            src = order[first]  # index (in this batch) of the best candidate for each touched niche
            better = f[src] > a_fit[uniq]
            j, s = uniq[better], src[better]
            a_fit[j] = f[s]
            a_tab[j] = np.asarray(tab)[s]
            a_wa[j] = np.asarray(wa)[s]
            a_wb[j] = np.asarray(wb)[s]

        (ki,) = self.split(1)
        tab, wa, wb = net.init_pop(ki, c.pop)
        (kb0,) = self.split(1)
        insert(tab, wa, wb, self.fitness(tab, wa, wb, kb0), self.desc(tab, wa, wb))
        for gen in range(c.gens):
            mut, wmut = self.rates(gen)
            kb, kp, km = self.split(3)
            occ = np.flatnonzero(np.isfinite(a_fit))
            pick = np.asarray(jax.random.randint(kp, (c.pop,), 0, len(occ)))
            sel = occ[pick]  # breed from random elites in the archive
            pt = jnp.asarray(a_tab[sel])
            pa, pb = jnp.asarray(a_wa[sel]), jnp.asarray(a_wb[sel])
            ct, ca, cb = mutate(km, net, pt, pa, pb, mut, wmut)  # mutation-only, as MAP-Elites is
            cfit = self.fitness(ct, ca, cb, kb)
            insert(ct, ca, cb, cfit, self.desc(ct, ca, cb))
            j = int(np.argmax(a_fit))
            self.cur = (jnp.asarray(a_tab[j]), jnp.asarray(a_wa[j]), jnp.asarray(a_wb[j]))
            self.log(gen, cfit, extra=f"filled {len(occ):3d}/{G * G}")

    # ---- nslc: novelty + beating your behavioural neighbours --------------------------
    def run_nslc(self):
        c, net = self.cfg, self.net
        (ki,) = self.split(1)
        tab, wa, wb = net.init_pop(ki, c.pop)
        arch = np.zeros((0, net.classes), np.float32)  # behaviour archive
        ar = jnp.arange(c.pop)
        for gen in range(c.gens):
            mut, wmut = self.rates(gen)
            kb, kt, kc, km, ka = self.split(5)
            fit = self.fitness(tab, wa, wb, kb)
            bh = np.asarray(self.behav(tab, wa, wb))  # (P, 10)
            pool = np.concatenate([arch, bh]) if len(arch) else bh
            d = np.linalg.norm(bh[:, None, :] - pool[None, :, :], axis=-1)  # (P, |pool|)
            k = min(c.knn, pool.shape[0] - 1)
            nn = np.argsort(d, 1)[:, 1 : k + 1]  # k nearest behaviours (excluding self)
            novelty = np.take_along_axis(d, nn, 1).mean(1)  # how unlike its neighbours it is
            f = np.asarray(fit)
            fpool = np.concatenate([np.full(len(arch), -np.inf, np.float32), f])
            local = (f[:, None] > fpool[nn]).mean(1)  # fraction of neighbours it beats
            z = lambda v: (v - v.mean()) / (v.std() + 1e-8)  # noqa: E731
            score = jnp.asarray(c.novelty_w * z(novelty) + z(local))  # the NSLC selection signal
            # archive a random slice of behaviours (standard novelty-archive growth), capped: the
            # k-NN distance matrix is (pop x |archive|) every iteration, so an unbounded archive
            # would dominate the runtime.
            keep = np.asarray(jax.random.uniform(ka, (c.pop,))) < 0.02
            arch = np.concatenate([arch, bh[keep]])[-4000:]
            i = int(jnp.argmax(fit))
            self.cur = (tab[i], wa[i], wb[i])
            self.log(gen, fit, extra=f"nov {novelty.mean():.3f} arch {len(arch)}")
            cand = jax.random.randint(kt, (c.pop, c.tsize), 0, c.pop)
            win = cand[ar, score[cand].argmax(1)]  # tournament on novelty+local-competition
            pt, pa, pb = tab[win], wa[win], wb[win]
            ct, ca, cb = crossover(kc, net, pt, pa, pb)
            tab, wa, wb = mutate(km, net, ct, ca, cb, mut, wmut)

    # ---- eda (UMDA/PBIL): carry a DISTRIBUTION over genomes, not a population ---------
    def run_eda(self):
        """Univariate EDA. The genome factorises exactly — a Bernoulli per table bit and a
        Categorical(K) per wire — so instead of shuffling genomes around we estimate, for every
        variable independently, the distribution that good genomes are drawn from.

        Each round: sample the population, keep the top `eda_elite`, and move the marginals toward the
        elites' statistics. No crossover, no mutation: the operators are replaced wholesale, which is
        the point — in ~28k dimensions, crossover propagates far less information than the marginals do.
        Marginals are clamped away from 0/1 so a bit can never lock in permanently.
        """
        c, net, K = self.cfg, self.net, self.cfg.wire_codebook
        n_el = max(2, int(c.pop * c.eda_elite))
        p = jnp.full((net.n_bits,), 0.5)  # P(table bit = 1)
        qa = jnp.full((net.n_gates, K), 1.0 / K)  # P(wire A takes codebook option k)
        qb = jnp.full((net.n_gates, K), 1.0 / K)
        for gen in range(c.gens):
            ks, kb = self.split(2)
            kt, ka, kbb = jax.random.split(ks, 3)
            tab = (jax.random.uniform(kt, (c.pop, net.n_bits)) < p).astype(jnp.uint8)
            la, lb = jnp.log(qa + 1e-9), jnp.log(qb + 1e-9)
            wa = jax.random.categorical(ka, jnp.broadcast_to(la, (c.pop, net.n_gates, K)), axis=-1)
            wb = jax.random.categorical(kbb, jnp.broadcast_to(lb, (c.pop, net.n_gates, K)), axis=-1)
            wa, wb = wa.astype(jnp.int32), wb.astype(jnp.int32)
            fit = self.fitness(tab, wa, wb, kb)
            el = jnp.argsort(-fit)[:n_el]  # truncation selection
            # move the marginals toward the elite statistics (PBIL-style learning rate)
            p_el = tab[el].mean(0)
            qa_el = jax.nn.one_hot(wa[el], K).mean(0)
            qb_el = jax.nn.one_hot(wb[el], K).mean(0)
            p = jnp.clip((1 - c.eda_lr) * p + c.eda_lr * p_el, c.eda_pmin, 1 - c.eda_pmin)
            qa = (1 - c.eda_lr) * qa + c.eda_lr * qa_el
            qb = (1 - c.eda_lr) * qb + c.eda_lr * qb_el
            qa = qa / qa.sum(-1, keepdims=True)
            qb = qb / qb.sum(-1, keepdims=True)
            # the deployable model is the MODE of the distribution; also check the best sample
            self.cur = (
                (p > 0.5).astype(jnp.uint8),
                qa.argmax(-1).astype(jnp.int32),
                qb.argmax(-1).astype(jnp.int32),
            )
            if gen % c.log_every == 0 or gen == c.gens - 1:
                self.test_best(tab[el[0]], wa[el[0]], wb[el[0]])  # best sampled genome
                conv = float(jnp.mean(jnp.abs(p - 0.5)) * 2)  # 0 = undecided, 1 = fully converged
                self.log(gen, fit, extra=f"conv {conv:.3f}")

    # ---- snes: estimate a gradient from the population, then step along it -------------
    def run_snes(self):
        """Separable Natural Evolution Strategies.

        Keeps a Gaussian N(mu, sigma) over CONTINUOUS latents that decode to the discrete genome
        (table bit = latent > 0; wire = argmax over its K codebook logits). Each round it perturbs the
        mean in `pop` random directions and forms a fitness-weighted average of those directions — an
        *estimated gradient* of expected fitness — then steps along it. Unlike the GA it never asks
        "who survived", it asks "which direction was uphill", which is what actually scales in ~92k
        dimensions. sigma is per-parameter (separable), so it stays O(d): full-covariance NES/CMA-ES
        would need a 92k x 92k matrix and is out of the question.

        Fitness is rank-shaped (standard NES utilities), so outliers and the noisy minibatch fitness
        cannot blow up a step.
        """
        c, net, K = self.cfg, self.net, self.cfg.wire_codebook
        nb, ng = net.n_bits, net.n_gates
        # ONE scalar latent per wire, binned monotonically into the K codebook options — NOT K
        # independent logits. K logits per wire would make d ~92k (80% of it redundant), and the ES
        # gradient estimate's variance grows with d, which drowns the signal.
        d = nb + 2 * ng
        mu = jnp.zeros(d)
        sigma = jnp.full(d, 1.0)
        lr_sigma = (3 + jnp.log(d)) / (5 * jnp.sqrt(d))  # standard SNES schedule

        # rank-based utilities: only the ORDER of fitnesses matters, never their scale
        r = jnp.arange(1, c.pop + 1)
        u = jnp.maximum(0.0, jnp.log(c.pop / 2 + 1) - jnp.log(r))
        u = u / u.sum() - 1.0 / c.pop

        def decode(x):
            """latents -> (tables, wire A choices, wire B choices).

            A table bit is the sign of its latent. A wire's latent is squashed and binned into one of
            the K codebook options, so nudging the latent moves the wire through neighbouring options
            instead of flipping it arbitrarily — ES needs a monotone, locally-smooth decode to follow.
            """
            tab = (x[:, :nb] > 0).astype(jnp.uint8)
            w = jax.nn.sigmoid(x[:, nb:])  # (P, 2*ng) in (0,1)
            ch = jnp.clip((w * K).astype(jnp.int32), 0, K - 1)
            return tab, ch[:, :ng], ch[:, ng:]

        for gen in range(c.gens):
            ke, kb = self.split(2)
            eps = jax.random.normal(ke, (c.pop, d))
            x = mu + sigma * eps
            tab, wa, wb = decode(x)
            fit = self.fitness(tab, wa, wb, kb)
            order = jnp.argsort(-fit)  # best first
            eps_s = eps[order]
            g_mu = jnp.einsum("i,ij->j", u, eps_s)  # the estimated natural gradient
            g_sig = jnp.einsum("i,ij->j", u, eps_s**2 - 1.0)
            mu = mu + c.snes_lr_mu * sigma * g_mu
            sigma = sigma * jnp.exp(0.5 * lr_sigma * g_sig)
            # the deployable model is the DECODED MEAN; also check the best sample
            self.cur = tuple(v[0] for v in decode(mu[None]))
            if gen % c.log_every == 0 or gen == c.gens - 1:
                i = int(order[0])
                self.test_best(tab[i], wa[i], wb[i])
                self.log(gen, fit, extra=f"sigma {float(sigma.mean()):.3f}")

    # ---- gomea: crossover with the linkage LEARNED, and every mix fitness-gated --------
    def run_gomea(self):
        """GOMEA / LTGA-style optimal mixing, adapted to the GPU.

        Plain crossover assumes the right inheritance unit is a single gate. GOMEA *learns* which
        genes belong together: gates whose values co-vary across the population are clustered into a
        linkage tree (UPGMA on a correlation proxy), and mixing copies a whole linked SUBSET from a
        random donor — kept only if fitness does not drop on the generation's batch (optimal mixing:
        every mix is fitness-gated, so bad recombinations never enter the population).

        Budget: each generation costs (1 + gomea_trials) pop-evaluations (one to score the current
        population, one per subset trial), so gens is scaled down to keep pop*gens*(T+1) equal to the
        other algorithms' budget. The linkage tree is rebuilt every gomea_rebuild gens; the gene-value
        correlation uses a fixed random embedding of each gate's (table, wireA, wireB) value — a cheap
        MI proxy that captures co-occurrence without a 4596^2 x 1024^2 joint histogram.
        """
        import scipy.cluster.hierarchy as sch

        c, net = self.cfg, self.net
        T = c.gomea_trials
        gens = max(1, c.gens // (T + 1))  # budget-matched
        (ki,) = self.split(1)
        tab, wa, wb = net.init_pop(ki, c.pop)
        K = net.codebook or 1
        (ke,) = self.split(1)
        embed = jax.random.normal(ke, (16 * K * K, 4))  # fixed random embedding of gene values

        def gene_vals(tab, wa, wb):  # (pop, n_gates) integer gene value per gate
            return net.codes(tab).astype(jnp.int32) * K * K + wa * K + wb

        def build_subsets(tab, wa, wb, kseed):
            """Linkage tree -> a bank of gene subsets (padded index arrays + lengths)."""
            gv = np.asarray(gene_vals(tab, wa, wb))  # (pop, G)
            E = np.asarray(embed)[gv].reshape(c.pop, -1, 4)  # (pop, G, 4)
            E = E - E.mean(0, keepdims=True)
            E = E / (np.linalg.norm(E, axis=0, keepdims=True) + 1e-9)
            # correlation proxy between gates: sum over embedding dims of corr^2
            R = np.einsum("pgd,phd->gh", E, E) ** 2  # (G, G) in [0, ~4]
            np.fill_diagonal(R, 0)
            dist = 1.0 / (R + 1e-3)  # co-varying gates = small distance
            iu = np.triu_indices(net.n_gates, 1)
            Z = sch.linkage(dist[iu], method="average")
            subsets = []
            members = {i: [i] for i in range(net.n_gates)}
            for j, (a, b, _, _) in enumerate(Z):
                m = members[int(a)] + members[int(b)]
                members[net.n_gates + j] = m
                if 2 <= len(m) <= c.gomea_max_subset:
                    subsets.append(m)
            rng = np.random.default_rng(kseed)
            rng.shuffle(subsets)
            return subsets

        subsets = build_subsets(tab, wa, wb, 0)
        for gen in range(gens):
            kb, kd, ks = self.split(3)
            if gen > 0 and gen % c.gomea_rebuild == 0:
                subsets = build_subsets(tab, wa, wb, gen)
            fit = self.fitness(tab, wa, wb, kb)  # everyone scored on THIS batch
            idx = jax.random.randint(kb, (c.batch,), 0, self.Xtr.shape[0])  # reuse kb's batch
            Xp, y = pack(self.Xtr[idx]), self.ytr[idx]
            pick = np.asarray(jax.random.randint(ks, (T,), 0, len(subsets)))
            for t in range(
                T
            ):  # optimal mixing: T fitness-gated donor copies, all individuals at once
                s = jnp.asarray(subsets[int(pick[t])])
                mask = jnp.zeros((net.n_gates,), bool).at[s].set(True)
                donor = jax.random.permutation(jax.random.fold_in(kd, t), c.pop)
                ct = jnp.where(jnp.repeat(mask, 4)[None, :], tab[donor], tab)
                ca = jnp.where(mask[None, :], wa[donor], wa)
                cb = jnp.where(mask[None, :], wb[donor], wb)
                margin, acc = net.eval_pop(net.codes(ct), ca, cb, Xp, y)
                cfit = margin + c.acc_weight * acc
                self.evals += c.pop
                ok = (cfit >= fit)[:, None]  # keep only non-worsening mixes
                tab = jnp.where(ok, ct, tab)
                wa, wb = jnp.where(ok, ca, wa), jnp.where(ok, cb, wb)
                fit = jnp.maximum(fit, cfit)
            i = int(jnp.argmax(fit))
            self.cur = (tab[i], wa[i], wb[i])
            if gen % max(1, c.log_every // (T + 1)) == 0 or gen == gens - 1:
                v = self.test_best(*self.cur_best())
                print(
                    f"gen {gen:5d}  fit best {float(jnp.max(fit)):8.2f}  mean {float(jnp.mean(fit)):8.2f}"
                    f"  TEST {v:.4f}  (best {self.best[1]:.4f})  subsets {len(subsets)}",
                    flush=True,
                )

    # ---- shared DID machinery for the ga couplings ------------------------------------
    def _did_jits(self):
        net = self.net
        return (
            jax.jit(functools.partial(did_forward_layers, net)),
            jax.jit(functools.partial(did_head, net)),
            jax.jit(functools.partial(did_influence, net)),
            [jax.jit(functools.partial(did_layer_c, net, li)) for li in range(len(net.widths))],
            jax.jit(functools.partial(did_trial_scan, net), static_argnames=("fit_w",)),
            jax.jit(functools.partial(did_fitness, net)),
            [jax.jit(functools.partial(did_rewire_c, net, li)) for li in range(len(net.widths))],
            jax.jit(functools.partial(did_rewire_trial_scan, net), static_argnames=("fit_w",)),
            [
                jax.jit(functools.partial(did_rewire_c_joint, net, li))
                for li in range(len(net.widths))
            ],
        )

    def _did_pass(self, jits, code, wa, wb, kb):
        """Forward + influence + C bins for ONE genome on a fresh batch (3-eval surcharge).

        Proposals always come from the CE surrogate; the returned anchor loss matches the burst's
        acceptance objective — CE, or the GA objective when `did_accept_fit` (fit-gated bursts)."""
        fwd, head, infl, layer_c, _, fitn, _, _, _ = jits
        c, net = self.cfg, self.net
        idx = jax.random.randint(kb, (c.batch,), 0, self.Xtr.shape[0])
        Xp, y = pack(self.Xtr[idx]), self.ytr[idx]
        acts = fwd(code, wa, wb, Xp)
        loss, lam_out = head(acts[-1], y)
        if c.did_accept_fit:
            margin, acc = fitn(acts[-1], y)
            loss = -(margin + c.acc_weight * acc)
        lams = infl(code, wa, wb, acts, Xp, lam_out)
        c_lay = [
            layer_c[li](lams[li], Xp if li == 0 else acts[li - 1], wa, wb)
            for li in range(len(net.widths))
        ]
        self.evals += 3
        return Xp, y, loss, jnp.concatenate(c_lay), c_lay, acts, lams

    def _did_burst(self, jits, code, wa, wb, kb):
        """One Lamarckian DID sweep on a single genome; returns the updated (code, wa, wb).

        Singleton table proposals always; with `did_rewire` the codebook rewire pool joins them
        (+32 eval surcharge for the scoring). Acceptance is strict on CE, or on the GA objective
        with `did_accept_fit`."""
        c, net = self.cfg, self.net
        Xp, y, loss, c_cat, c_lay, acts, lams = self._did_pass(jits, code, wa, wb, kb)
        pcode, delta = did_best_response(c_cat, code)
        if c.did_rewire:
            rew, rewj = jits[6], jits[8]
            cab = [
                rew[li](lams[li], Xp if li == 0 else acts[li - 1], wa, wb)
                for li in range(len(net.widths))
            ]
            cjs = (
                [
                    rewj[li](lams[li], Xp if li == 0 else acts[li - 1])
                    for li in range(len(net.widths))
                ]
                if c.did_joint
                else None
            )
            g1, t1, g2, t2, p1, k1, dl = did_rewire_pool(
                net, code, wa, wb, pcode, delta, cab, c_lay, cjs
            )
            self.evals += 32 + (80 if c.did_joint else 0)
        else:
            g1 = np.arange(net.n_gates, dtype=np.int32)
            t1, dl = np.asarray(pcode), np.asarray(delta)
            g2, t2 = g1, t1
            p1 = np.full(net.n_gates, -1, np.int32)
            k1 = np.zeros(net.n_gates, np.int32)
        neg = np.flatnonzero(dl < 0)
        order = neg[np.argsort(dl[neg])][: c.did_props]
        n = len(order)
        if n == 0:
            return code, wa, wb
        P = c.did_props

        def takev(v):
            return jnp.asarray(np.concatenate([v[order], np.zeros(P - n, v.dtype)]))

        valid = jnp.asarray(np.arange(P) < n)
        go, po, ko = g1[order], p1[order], k1[order]
        sna, snb = did_rewire_snew(net, go, po, ko)
        sa_all, sb_all = net.sources(wa, wb)
        fit_w = float(c.acc_weight) if c.did_accept_fit else None
        code, _, _, _, n_tr, _, accepted = jits[7](
            code,
            sa_all,
            sb_all,
            Xp,
            y,
            takev(g1),
            takev(t1),
            takev(g2),
            takev(t2),
            takev(p1),
            jnp.asarray(np.concatenate([sna, np.zeros(P - n, np.int32)])),
            jnp.asarray(np.concatenate([snb, np.zeros(P - n, np.int32)])),
            valid,
            loss,
            fit_w=fit_w,
        )
        wa, wb = did_rewire_apply(net, wa, wb, go, po, ko, np.asarray(accepted)[:n])
        self.evals += int(n_tr)
        return code, wa, wb

    # ---- did: C-driven proposals, global priority, exact one-at-a-time acceptance -----
    def run_did(self):
        """Core Discrete Influence Descent — ONE network, no population.

        Per sweep (one fresh `batch`, the same size every other algorithm's fitness uses):
        forward + closed-form output influence + signed-sensitivity backprop give every gate's
        C bins. Singleton proposals flip only C-driven bits (C_p == 0 keeps the current bit);
        `did_parent_child` adds counterfactual parent-child bundles, `did_rewire` adds per-port
        codebook rewires with their joint best-response tables (`did_joint`: also all K^2
        two-port candidate pairs) — all in the same pool. Every
        proposal with a strictly negative surrogate delta is ranked GLOBALLY by that delta, and
        the top `did_props` are evaluated ONE AT A TIME by exact full forward on the same batch,
        accepted only if the real loss strictly falls. Influence and proposals are computed once
        per sweep (fixed linearisation); a proposal made stale by an earlier accept in the same
        sweep just fails its trial.

        Budget: every exact trial is one network evaluation on `batch` samples — the same
        currency the other algorithms are charged in — plus a per-sweep surcharge for the
        scoring (forward + influence + C, the motif rollouts, the curvature pass, the rewire
        einsums); sweeps run until the shared pop*gens budget is spent.
        """
        c, net = self.cfg, self.net
        L, P = len(net.widths), c.did_props
        if c.did_rewire:
            assert not (c.did_order2 or c.did_ema > 0 or c.did_confirm or c.did_accept_t0 > 0), (
                "--did-rewire composes with --did-accept-fit and --did-parent-child only"
            )
            assert net.codebook, "--did-rewire needs codebook wiring"
        assert not (c.did_joint and not c.did_rewire), "--did-joint requires --did-rewire"
        assert not (c.distill_propose_only and c.distill_accept_only), (
            "--distill-propose-only and --distill-accept-only are exclusive"
        )
        assert not ((c.distill_propose_only or c.distill_accept_only) and not c.distill), (
            "the distill mechanism splits need --distill"
        )
        assert not (c.distill and c.did_confirm), "--distill does not compose with --did-confirm"
        tgt_all = self.tgt_all
        fwd = jax.jit(functools.partial(did_forward_layers, net))
        head = jax.jit(functools.partial(did_head, net))
        infl = jax.jit(functools.partial(did_influence, net))
        layer_c = [jax.jit(functools.partial(did_layer_c, net, li)) for li in range(L)]
        motif = [jax.jit(functools.partial(did_motif_props, net, li)) for li in range(L)]
        trial = jax.jit(functools.partial(did_trial_scan, net), static_argnames=("fit_w",))
        rew = [jax.jit(functools.partial(did_rewire_c, net, li)) for li in range(L)]
        rewj = [jax.jit(functools.partial(did_rewire_c_joint, net, li)) for li in range(L)]
        rtrial = jax.jit(functools.partial(did_rewire_trial_scan, net), static_argnames=("fit_w",))
        head_gamma = jax.jit(functools.partial(did_head_gamma, net))
        curv = jax.jit(functools.partial(did_curvature, net))
        fitn = jax.jit(functools.partial(did_fitness, net))
        fit_w = float(c.acc_weight) if c.did_accept_fit else None

        def anchor(acts_last, yy, ce_loss):
            """The trial scan's accepted quantity for the current genome on one batch."""
            if fit_w is None:
                return ce_loss
            margin, a = fitn(acts_last, yy)
            return -(margin + fit_w * a)

        (ki,) = self.split(1)
        tab, wa, wb = net.init_pop(ki, 1)
        if c.init_genome:
            gt, gwa, gwb = load_genome(c.init_genome)
            tab, wa, wb = gt[None], gwa[None], gwb[None]
        code, wa, wb = net.codes(tab)[0], wa[0], wb[0]
        target = c.pop * c.gens
        surcharge = (
            3
            + (16 if c.did_parent_child else 0)
            + (2 if c.did_order2 else 0)
            + (32 if c.did_rewire else 0)  # 4 einsums of N*K*w across ports ~ 32 packed forwards
            + (80 if c.did_joint else 0)  # the N*K^2*w triple contraction ~ 64 forwards + sums
        )
        c1_bar = c2_bar = None  # EMA state (coefficients from earlier sweeps' genomes: momentum)
        sweep_i = 0
        while self.evals < target:
            if c.did_confirm:
                kb, kb2 = self.split(2)
            else:
                (kb,) = self.split(1)
            idx = jax.random.randint(kb, (c.batch,), 0, self.Xtr.shape[0])
            Xp, y = pack(self.Xtr[idx]), self.ytr[idx]
            tgt = tgt_all[idx] if tgt_all is not None else None
            t_prop = None if c.distill_accept_only else tgt
            t_acc = None if c.distill_propose_only else tgt
            acts = fwd(code, wa, wb, Xp)
            loss, lam_out = head(acts[-1], y, t_prop)
            if t_acc is not t_prop:
                loss = head(acts[-1], y, t_acc)[0]  # anchor on the acceptance objective
            lams = infl(code, wa, wb, acts, Xp, lam_out)
            c_all = [
                layer_c[li](lams[li], Xp if li == 0 else acts[li - 1], wa, wb) for li in range(L)
            ]
            c1_cat, c2_cat = jnp.concatenate(c_all), None
            if c.did_order2:
                gams = curv(code, wa, wb, acts, Xp, head_gamma(acts[-1], y))
                c2_cat = jnp.concatenate(
                    [
                        layer_c[li](gams[li], Xp if li == 0 else acts[li - 1], wa, wb)
                        for li in range(L)
                    ]
                )
            if c.did_ema > 0:
                b = c.did_ema
                c1_bar = c1_cat if c1_bar is None else b * c1_bar + (1 - b) * c1_cat
                c1_cat = c1_bar
                if c2_cat is not None:
                    c2_bar = c2_cat if c2_bar is None else b * c2_bar + (1 - b) * c2_cat
                    c2_cat = c2_bar
            score = did_effective_c(c1_cat, c2_cat, code) if c.did_order2 else c1_cat
            pcode, delta = did_best_response(score, code)
            gi = jnp.arange(net.n_gates, dtype=jnp.int32)
            obj = anchor(acts[-1], y, loss)
            loss0 = float(obj)
            if c.did_rewire:
                cab = [
                    rew[li](lams[li], Xp if li == 0 else acts[li - 1], wa, wb) for li in range(L)
                ]
                cjs = (
                    [rewj[li](lams[li], Xp if li == 0 else acts[li - 1]) for li in range(L)]
                    if c.did_joint
                    else None
                )
                mo = None
                if c.did_parent_child:
                    mg = [[], [], [], [], []]
                    for li in range(1, L):
                        pin = Xp if li == 1 else acts[li - 2]
                        m = motif[li](lams[li], pin, acts[li - 1], code, wa, wb, c_all[li - 1])
                        for lst, v in zip(mg, m):
                            lst.append(v)
                    mo = tuple(jnp.concatenate(v) for v in mg)
                g1, t1, g2, t2, p1, k1, dl = did_rewire_pool(
                    net, code, wa, wb, pcode, delta, cab, c_all, cjs, mo
                )
                is_aux, aux_lab = p1 >= 0, "rewire"
                neg = np.flatnonzero(dl < 0)
                order = neg[np.argsort(dl[neg])]
                order = did_dedup_order(order, g1, g2, P) if c.did_dedup else order[:P]
                n = len(order)

                def take(v):
                    return jnp.asarray(np.concatenate([v[order], np.zeros(P - n, v.dtype)]))

                valid = jnp.asarray(np.arange(P) < n)
                go, po, ko = g1[order], p1[order], k1[order]
                sna, snb = did_rewire_snew(net, go, po, ko)
                sa_all, sb_all = net.sources(wa, wb)
                code, _, _, obj, n_tr, n_ac, accepted = rtrial(
                    code,
                    sa_all,
                    sb_all,
                    Xp,
                    y,
                    take(g1),
                    take(t1),
                    take(g2),
                    take(t2),
                    take(p1),
                    jnp.asarray(np.concatenate([sna, np.zeros(P - n, np.int32)])),
                    jnp.asarray(np.concatenate([snb, np.zeros(P - n, np.int32)])),
                    valid,
                    obj,
                    fit_w=fit_w,
                    targets=t_acc,
                )
                wa, wb = did_rewire_apply(net, wa, wb, go, po, ko, np.asarray(accepted)[:n])
                self.evals += int(n_tr) + surcharge
            else:
                g1, c1, g2, c2, dl = [gi], [pcode], [gi], [pcode], [delta]
                if c.did_parent_child:
                    for li in range(1, L):
                        pin = Xp if li == 1 else acts[li - 2]
                        m = motif[li](lams[li], pin, acts[li - 1], code, wa, wb, c_all[li - 1])
                        for lst, v in zip((g1, c1, g2, c2, dl), m):
                            lst.append(v)
                g1, c1, g2, c2, dl = (np.asarray(jnp.concatenate(v)) for v in (g1, c1, g2, c2, dl))
                is_aux, aux_lab = np.arange(len(dl)) >= net.n_gates, "motif"
                neg = np.flatnonzero(dl < 0)
                # global priority: most negative first
                order = neg[np.argsort(dl[neg])]
                order = did_dedup_order(order, g1, g2, P) if c.did_dedup else order[:P]
                n = len(order)

                def take(v):
                    return jnp.asarray(np.concatenate([v[order], np.zeros(P - n, v.dtype)]))

                valid = jnp.asarray(np.arange(P) < n)
                unif, temp = None, 0.0
                if c.did_accept_t0 > 0:
                    (ku,) = self.split(1)
                    unif = jax.random.uniform(ku, (P,))
                    temp = c.did_accept_t0 * 1e-3 ** min(self.evals / target, 1.0)
                Xp2 = y2 = obj2 = None
                if c.did_confirm:
                    idx2 = jax.random.randint(kb2, (c.batch,), 0, self.Xtr.shape[0])
                    Xp2, y2 = pack(self.Xtr[idx2]), self.ytr[idx2]
                    acts2_last = fwd(code, wa, wb, Xp2)[-1]
                    obj2 = anchor(acts2_last, y2, head(acts2_last, y2)[0])
                code, obj, n_tr, n_ac, accepted = trial(
                    code,
                    wa,
                    wb,
                    Xp,
                    y,
                    take(g1),
                    take(c1),
                    take(g2),
                    take(c2),
                    valid,
                    obj,
                    unif,
                    temp,
                    Xp2,
                    y2,
                    obj2,
                    fit_w=fit_w,
                    targets=t_acc,
                )
                self.evals += (
                    int(n_tr) * (2 if c.did_confirm else 1) + surcharge + int(c.did_confirm)
                )
            loss = obj  # the carried objective (CE, or -(margin + w*acc) in fit mode)
            self.cur = (tab_from_code(code), wa, wb)
            if sweep_i % 25 == 0 or self.evals >= target:
                sel_m = np.concatenate([is_aux[order], np.zeros(P - n, bool)])
                m_tr, m_ac = int(sel_m.sum()), int(np.asarray(accepted)[sel_m].sum())
                v = self.test_best(*self.cur)
                print(
                    f"sweep {sweep_i:5d}  loss {loss0:.4f} -> {float(loss):.4f}  "
                    f"acc {int(n_ac)}/{int(n_tr)} ({aux_lab} {m_ac}/{m_tr})  "
                    f"evals {self.evals / 1e6:5.2f}M  TEST {v:.4f}  (best {self.best[1]:.4f})",
                    flush=True,
                )
            sweep_i += 1

    # ---- hc: random-proposal control for did — same exact acceptance, no influence ----
    def run_hc(self):
        """Random hill climber — DID's acceptance regime with uniform random proposals.

        Per sweep (one fresh `batch`): `did_props` random single-gate table rewrites (uniform
        gate, uniform new table != current), tried ONE AT A TIME through the same trial scan
        and accepted only if the real loss strictly falls. No influence, no priority — this
        isolates DID's proposal QUALITY (hc shares its regime) from its acceptance REGIME
        (the ga shares neither). Budget: each trial is 1 evaluation, +1/sweep for the loss
        anchor forward.
        """
        c, net = self.cfg, self.net
        P = c.did_props
        fwd = jax.jit(functools.partial(did_forward_layers, net))
        head = jax.jit(functools.partial(did_head, net))
        trial = jax.jit(functools.partial(did_trial_scan, net))
        (ki,) = self.split(1)
        tab, wa, wb = net.init_pop(ki, 1)
        if c.init_genome:
            gt, gwa, gwb = load_genome(c.init_genome)
            tab, wa, wb = gt[None], gwa[None], gwb[None]
        code, wa, wb = net.codes(tab)[0], wa[0], wb[0]
        target = c.pop * c.gens
        valid = jnp.ones(P, bool)
        sweep_i = 0
        while self.evals < target:
            kb, kg, kt = self.split(3)
            idx = jax.random.randint(kb, (c.batch,), 0, self.Xtr.shape[0])
            Xp, y = pack(self.Xtr[idx]), self.ytr[idx]
            loss, _ = head(fwd(code, wa, wb, Xp)[-1], y)
            loss0 = float(loss)
            g = jax.random.randint(kg, (P,), 0, net.n_gates)
            step = jax.random.randint(kt, (P,), 1, 16)  # uniform over the 15 OTHER tables
            t = ((code[g].astype(jnp.int32) + step) % 16).astype(jnp.uint8)
            code, loss, n_tr, n_ac, _ = trial(code, wa, wb, Xp, y, g, t, g, t, valid, loss)
            self.evals += int(n_tr) + 1
            self.cur = (tab_from_code(code), wa, wb)
            if sweep_i % 25 == 0 or self.evals >= target:
                v = self.test_best(*self.cur)
                print(
                    f"sweep {sweep_i:5d}  loss {loss0:.4f} -> {float(loss):.4f}  "
                    f"acc {int(n_ac)}/{int(n_tr)}  "
                    f"evals {self.evals / 1e6:5.2f}M  TEST {v:.4f}  (best {self.best[1]:.4f})",
                    flush=True,
                )
            sweep_i += 1

    def cur_best(self):
        return self.cur

    def run(self):
        return {
            "ga": self.run_ga,
            "aging": self.run_aging,
            "mapelites": self.run_mapelites,
            "nslc": self.run_nslc,
            "eda": self.run_eda,
            "snes": self.run_snes,
            "gomea": self.run_gomea,
            "did": self.run_did,
            "hc": self.run_hc,
        }[self.cfg.algo]()


def did_selftest() -> None:
    """Unit checks for the DID implementation, on a tiny synthetic net (CPU, seconds).

    Verifies, in order: the layer-cached forward matches Net._forward; the closed-form output
    influence matches autodiff through a float head; the influence backprop matches a per-gate
    reference loop; the C bins satisfy their defining identity sum_p C_p t_p == sum_b lambda
    h(a_b, b_b) for random tables; the best response minimises the surrogate over all 16 tables
    and never flips a zero-evidence bit; parent-child motif proposals match a brute-force
    counterfactual recompute and their argmin; a priority-ordered trial scan never increases the
    batch loss and its carried loss matches a from-scratch forward of the final genome; and
    tab_from_code round-trips through Net.codes. Behavioural guards: C is invariant under batch
    duplication (sum-vs-mean scale regressions); the surrogate ranking of layer-0 proposals
    beats random on EXACT deltas (catches convention-shared corruption a reference-loop
    comparison would miss); and fixed-batch sweeps descend monotonically to a zero-accept
    plateau.
    """
    N, n_in, widths, classes, K = 64, 16, [12, 8], 2, 3
    group = widths[-1] // classes
    net = Net(n_in, widths, classes, codebook=K)
    kx, ky, ki = jax.random.split(jax.random.PRNGKey(0), 3)
    X = jax.random.bernoulli(kx, 0.5, (N, n_in)).astype(jnp.uint8)
    y = jax.random.randint(ky, (N,), 0, classes)
    Xp = pack(X)
    tab, wa, wb = net.init_pop(ki, 1)
    code, wa, wb = net.codes(tab)[0], wa[0], wb[0]

    # (0) forward with per-layer caching == Net._forward logits
    acts = did_forward_layers(net, code, wa, wb, Xp)
    logits_ref = net._forward(code, wa, wb, Xp)
    bits = jnp.unpackbits(acts[-1], axis=1)
    assert (bits.reshape(classes, group, N).sum(1) == logits_ref).all(), "forward mismatch"

    # (1) closed-form output influence == autodiff through the float head
    loss, lam_out = did_head(net, acts[-1], y)
    bf = unpack_rows(acts[-1])  # (N, w_out)

    def loss_bits(b):
        z = b.reshape(N, classes, group).sum(-1) / group**0.5
        return (jax.scipy.special.logsumexp(z, 1) - z[jnp.arange(N), y]).mean()

    assert abs(float(loss_bits(bf)) - float(loss)) < 1e-6, "head loss mismatch"
    g = jax.grad(loss_bits)(bf)
    assert np.allclose(np.asarray(g), np.asarray(lam_out), atol=1e-6), "output influence"

    # (2) influence backprop == per-gate reference loop (signed dh/dinput, not the XOR mask)
    lams = did_influence(net, code, wa, wb, acts, Xp, lam_out)
    af = [np.asarray(unpack_rows(a)) for a in acts]
    lam_ref = [np.zeros((N, w), np.float32) for w in widths]
    lam_ref[-1] = np.asarray(lam_out)
    for li in range(len(widths) - 1, 0, -1):
        lo = net.offs[li]
        sa, sb = (np.asarray(v) for v in did_resolve(net, wa, wb, lo, net.offs[li + 1]))
        cl = np.asarray(code[lo : net.offs[li + 1]])
        for i in range(widths[li]):
            h = lambda a_, b_: (int(cl[i]) >> (int(a_) * 2 + int(b_))) & 1  # noqa: E731
            for s in range(N):
                a_, b_ = af[li - 1][s, sa[i]], af[li - 1][s, sb[i]]
                lam_ref[li - 1][s, sa[i]] += lam_ref[li][s, i] * (h(1, b_) - h(0, b_))
                lam_ref[li - 1][s, sb[i]] += lam_ref[li][s, i] * (h(a_, 1) - h(a_, 0))
    for li in range(len(widths)):
        assert np.allclose(np.asarray(lams[li]), lam_ref[li], atol=1e-5), f"influence layer {li}"

    # (3) C bin identity: sum_p C_p t_p == sum_b lambda_j h(a_b, b_b) for random tables
    rng = np.random.default_rng(0)
    c_all = [
        did_layer_c(net, li, lams[li], Xp if li == 0 else acts[li - 1], wa, wb)
        for li in range(len(widths))
    ]
    for li in range(len(widths)):
        prev_p = Xp if li == 0 else acts[li - 1]
        Pf, lam_l, C = np.asarray(unpack_rows(prev_p)), np.asarray(lams[li]), np.asarray(c_all[li])
        sa, sb = (np.asarray(v) for v in did_resolve(net, wa, wb, net.offs[li], net.offs[li + 1]))
        for _ in range(20):
            j, newc = int(rng.integers(widths[li])), int(rng.integers(16))
            av, bv = Pf[:, sa[j]], Pf[:, sb[j]]
            h = np.array(
                [(newc >> (int(a_) * 2 + int(b_))) & 1 for a_, b_ in zip(av, bv)], np.float32
            )
            lhs = sum(C[j, p] * ((newc >> p) & 1) for p in range(4))
            assert abs(float(lhs) - float((lam_l[:, j] * h).sum())) < 1e-5, "C identity"

    # (3b) ground truth: C == dLoss/d(table bit) of the multilinear relaxation of the WHOLE
    # circuit (each gate equals its multilinear extension at {0,1} corners, so the relaxed
    # gradient at the corner IS the exact first-order coefficient). This validates lambda, its
    # propagation, and the binning end-to-end against autodiff — no shared convention with the
    # implementation.
    Xf = X.astype(jnp.float32)

    def ml_loss(tb_layers):
        prev = Xf
        for mli in range(len(widths)):
            msa, msb = did_resolve(net, wa, wb, net.offs[mli], net.offs[mli + 1])
            a_, b_ = prev[:, msa], prev[:, msb]
            t = tb_layers[mli]  # (w, 4), pattern p = 2a + b
            prev = (
                t[:, 0] * (1 - a_) * (1 - b_)
                + t[:, 1] * (1 - a_) * b_
                + t[:, 2] * a_ * (1 - b_)
                + t[:, 3] * a_ * b_
            )
        z = prev.reshape(N, classes, group).sum(-1) / group**0.5
        return (jax.scipy.special.logsumexp(z, 1) - z[jnp.arange(N), y]).mean()

    tb_layers = [code_bits(code[net.offs[i] : net.offs[i + 1]]) for i in range(len(widths))]
    assert abs(float(ml_loss(tb_layers)) - float(loss)) < 1e-6, "multilinear corner mismatch"
    g_ml = jax.grad(ml_loss)(tb_layers)
    for gli in range(len(widths)):
        assert np.allclose(np.asarray(g_ml[gli]), np.asarray(c_all[gli]), atol=1e-5), (
            f"C vs multilinear autodiff, layer {gli}"
        )

    # (4) best response: minimises the surrogate over all 16 tables, flips no zero-evidence bit
    pcode, delta = did_best_response(jnp.concatenate(c_all), code)
    pc_np, dl_np, C_np = np.asarray(pcode), np.asarray(delta), np.asarray(jnp.concatenate(c_all))
    g_np = np.asarray(code)
    tbl = np.array([[(h >> p) & 1 for p in range(4)] for h in range(16)], np.float32)  # (16, 4)
    gb = tbl[g_np]  # (n_gates, 4) current bits
    all_d = ((tbl[None] - gb[:, None]) * C_np[:, None, :]).sum(-1)  # (n_gates, 16)
    assert np.allclose(dl_np, all_d.min(1), atol=1e-6), "best response not minimal"
    flips = tbl[pc_np] != gb
    assert (np.abs(C_np)[flips] > 0).all(), "zero-evidence bit flipped"
    assert ((dl_np < 0) == (pc_np != g_np)).all(), "delta sign vs change mismatch"

    # (5) parent-child motifs: bins, best response, and argmin match a brute-force recompute
    li = 1
    lo_c, lo_p = net.offs[li], net.offs[li - 1]
    w_c = widths[li]
    m_g1, m_c1, m_g2, m_c2, m_d = (
        np.asarray(v)
        for v in did_motif_props(net, li, lams[li], Xp, acts[0], code, wa, wb, c_all[0])
    )
    csa, csb = (np.asarray(v) for v in did_resolve(net, wa, wb, lo_c, net.offs[li + 1]))
    p_in = np.asarray(unpack_rows(Xp))
    sa_p, sb_p = (np.asarray(v) for v in did_resolve(net, wa, wb, lo_p, net.offs[li]))
    a0 = np.asarray(unpack_rows(acts[0]))  # (N, w_p) current parent-layer outputs
    lam_c, C_par = np.asarray(lams[li]), np.asarray(c_all[0])
    for e in rng.choice(2 * w_c, 8, replace=False):
        slot_a, k = e < w_c, int(e % w_c)
        if csa[k] == csb[k]:
            assert np.isinf(m_d[e]), "degenerate edge not masked"
            continue
        jp = int(csa[k] if slot_a else csb[k])  # parent local index
        r = a0[:, csb[k]] if slot_a else a0[:, csa[k]]  # child's other input, current world
        gk = int(g_np[lo_c + k])
        d_bf = np.zeros(16)
        best_child = np.zeros(16, np.uint8)
        for h in range(16):
            z = tbl[h][(2 * p_in[:, sa_p[jp]] + p_in[:, sb_p[jp]]).astype(int)]  # counterfactual
            ab = (z, r) if slot_a else (r, z)
            bins = np.array(  # pattern p = 2*a + b
                [
                    (lam_c[:, k] * ((ab[0] == pa) & (ab[1] == pb))).sum()
                    for pa in (0, 1)
                    for pb in (0, 1)
                ]
            )
            tk = np.where(bins < 0, 1.0, np.where(bins > 0, 0.0, tbl[gk]))
            d_child = ((tk - tbl[gk]) * bins).sum()
            d_par = ((tbl[h] - tbl[int(g_np[lo_p + jp])]) * C_par[jp]).sum()
            d_bf[h] = d_par + d_child
            best_child[h] = int((tk * (2 ** np.arange(4))).sum())
        d_bf[int(g_np[lo_p + jp])] = np.inf  # h_j == g_j is the child's singleton — masked
        h_star = int(m_c1[e])
        assert m_g1[e] == lo_p + jp and m_g2[e] == lo_c + k, "motif indices"
        assert abs(m_d[e] - d_bf.min()) < 1e-4, "motif delta not minimal"
        assert abs(m_d[e] - d_bf[h_star]) < 1e-4, "motif delta mismatch at chosen parent"
        assert int(m_c2[e]) == int(best_child[h_star]), "motif child best response"

    # (6) trial scan: monotone loss, and the carried loss matches a from-scratch forward
    neg = np.flatnonzero(dl_np < 0)
    order = neg[np.argsort(dl_np[neg])][:32]
    n = len(order)
    assert n > 0, "no proposals at random init"
    gi = np.arange(net.n_gates, dtype=np.int32)
    pad = 32 - n

    def take(v):
        return jnp.asarray(np.concatenate([v[order], np.zeros(pad, v.dtype)]))

    code2, loss2, n_tr, n_ac, acc_mask = did_trial_scan(
        net,
        code,
        wa,
        wb,
        Xp,
        y,
        take(gi),
        take(pc_np),
        take(gi),
        take(pc_np),
        jnp.asarray(np.arange(32) < n),
        loss,
    )
    assert float(loss2) <= float(loss) + 1e-7, "trial scan increased the loss"
    assert int(n_tr) == n and 0 <= int(n_ac) <= n, "trial counts"
    assert int(np.asarray(acc_mask).sum()) == int(n_ac), "accept mask disagrees with count"
    fresh = did_forward_layers(net, code2, wa, wb, Xp)
    assert abs(float(did_head(net, fresh[-1], y)[0]) - float(loss2)) < 1e-5, "loss drifted"
    changed = np.flatnonzero(np.asarray(code2) != g_np)
    assert set(changed) <= set(gi[order].tolist()), "scan changed an unproposed gate"

    # (7) tab_from_code round-trips
    assert (net.codes(tab_from_code(code)[None])[0] == code).all(), "tab_from_code"

    # (8) batch-scale invariance: duplicating the batch must leave loss and C unchanged (lambda
    # carries the 1/n of the mean loss). Catches any future sum-vs-mean scale regression.
    Xp2, y2 = pack(jnp.concatenate([X, X])), jnp.concatenate([y, y])
    acts2 = did_forward_layers(net, code, wa, wb, Xp2)
    loss2x, lam2 = did_head(net, acts2[-1], y2)
    assert abs(float(loss2x) - float(loss)) < 1e-6, "loss not batch-scale invariant"
    lams2 = did_influence(net, code, wa, wb, acts2, Xp2, lam2)
    for li in range(len(widths)):
        C2 = did_layer_c(net, li, lams2[li], Xp2 if li == 0 else acts2[li - 1], wa, wb)
        assert np.allclose(np.asarray(C2), np.asarray(c_all[li]), atol=1e-6), (
            f"C not batch-scale invariant, layer {li}"
        )

    # (9) ranking calibration: top-ranked proposals must mostly improve the EXACT loss and
    # enrich it far beyond uniform random proposals. No convention is shared with the
    # implementation — only end-to-end usefulness is asserted, so a corrupted backward pass
    # (e.g. an unsigned sensitivity) fails here even if it passes every reference comparison.
    def exact_delta(g, t):
        fresh_c = did_forward_layers(net, code.at[g].set(t), wa, wb, Xp)
        return float(did_head(net, fresh_c[-1], y)[0]) - float(loss)

    lay0 = order[order < widths[0]]  # ranked layer-0 proposals: only these need PROPAGATED lambda
    assert len(lay0) >= 4, "too few ranked layer-0 proposals to calibrate"
    d_top = np.array([exact_delta(int(g), np.uint8(pc_np[g])) for g in lay0])
    rngc = np.random.default_rng(1)
    d_rand = np.array(
        [
            exact_delta(int(g), np.uint8((int(g_np[g]) + int(rngc.integers(1, 16))) % 16))
            for g in rngc.integers(0, net.n_gates, 64)
        ]
    )
    # measured on this fixed seed: signed 0.57 / mean -0.006; unsigned sensitivity 0.29 /
    # +0.018; uniform random 0.25 / +0.013 — the thresholds separate correct from corrupt
    frac = float((d_top < 0).mean())
    assert frac >= 0.5, f"top layer-0 proposals mostly fail exactly ({frac:.2f} improve)"
    assert d_top.mean() < 0, f"top layer-0 proposals don't improve on average ({d_top.mean():.5f})"
    assert d_top.mean() < d_rand.mean() - 1e-3, (
        f"ranking no better than random (top {d_top.mean():.5f} vs rand {d_rand.mean():.5f})"
    )

    # (10) fixed-batch descent reaches a zero-accept plateau: sweeping propose->trial on ONE
    # batch is monotone across sweeps and terminates — once a sweep accepts nothing the state
    # repeats exactly, so zero accepts IS the fixed point.
    code_p, prev_loss = code, float(loss)
    for _ in range(60):
        acts_p = did_forward_layers(net, code_p, wa, wb, Xp)
        l_p, lam_p = did_head(net, acts_p[-1], y)
        assert float(l_p) <= prev_loss + 1e-6, "fixed-batch sweep increased the loss"
        prev_loss = float(l_p)
        lams_p = did_influence(net, code_p, wa, wb, acts_p, Xp, lam_p)
        c_p = jnp.concatenate(
            [
                did_layer_c(net, li, lams_p[li], Xp if li == 0 else acts_p[li - 1], wa, wb)
                for li in range(len(widths))
            ]
        )
        prop_p, d_p = did_best_response(c_p, code_p)
        negp = np.flatnonzero(np.asarray(d_p) < 0)
        if len(negp) == 0:
            break
        ordp = negp[np.argsort(np.asarray(d_p)[negp])][:32].astype(np.int32)
        pad_p = 32 - len(ordp)
        gseq = jnp.asarray(np.concatenate([ordp, np.zeros(pad_p, np.int32)]))
        tseq = jnp.asarray(np.concatenate([np.asarray(prop_p)[ordp], np.zeros(pad_p, np.uint8)]))
        vld = jnp.asarray(np.arange(32) < len(ordp))
        code_p, l_p, _, n_ac_p, _ = did_trial_scan(
            net, code_p, wa, wb, Xp, y, gseq, tseq, gseq, tseq, vld, l_p
        )
        prev_loss = float(l_p)
        if int(n_ac_p) == 0:
            break
    else:
        raise AssertionError("no zero-accept plateau within 60 fixed-batch sweeps")

    # (11) head curvature == the exact diagonal of the head Hessian w.r.t. output bits
    gam_out = did_head_gamma(net, acts[-1], y)
    H = jax.hessian(loss_bits)(bf)  # (N, w_out, N, w_out)
    diag = jnp.einsum("nwnw->nw", H)
    assert np.allclose(np.asarray(diag), np.asarray(gam_out), atol=1e-6), "head curvature"

    # (12) second-order machinery: C2 bins are nonnegative, the effective-coefficient best
    # response minimises the QUADRATIC surrogate over all 16 tables, and curvature only ever
    # damps (its flip set is a subset of the first-order flip set)
    gams = did_curvature(net, code, wa, wb, acts, Xp, gam_out)
    c2_all = jnp.concatenate(
        [
            did_layer_c(net, li, gams[li], Xp if li == 0 else acts[li - 1], wa, wb)
            for li in range(len(widths))
        ]
    )
    C2n = np.asarray(c2_all)
    assert (C2n >= -1e-9).all(), "curvature bins negative"
    p2c, d2c = did_best_response(did_effective_c(jnp.concatenate(c_all), c2_all, code), code)
    dt = tbl[None] - gb[:, None]  # (n_gates, 16, 4)
    q = (dt * C_np[:, None]).sum(-1) + 0.5 * ((dt**2) * C2n[:, None]).sum(-1)
    assert np.allclose(np.asarray(d2c), q.min(1), atol=1e-5), "order-2 best response not minimal"
    assert np.allclose(q[np.arange(len(q)), np.asarray(p2c)], np.asarray(d2c), atol=1e-5), (
        "order-2 delta mismatch at chosen table"
    )
    flips2 = tbl[np.asarray(p2c)] != gb
    assert not (flips2 & ~(tbl[pc_np] != gb)).any(), "curvature added a flip"

    # (13) confirm-mode trial scan with the SAME batch as confirm batch == plain scan
    code3, _, n_tr3, n_ac3, _ = did_trial_scan(
        net,
        code,
        wa,
        wb,
        Xp,
        y,
        take(gi),
        take(pc_np),
        take(gi),
        take(pc_np),
        jnp.asarray(np.arange(32) < n),
        loss,
        None,
        0.0,
        Xp,
        y,
        loss,
    )
    assert (np.asarray(code3) == np.asarray(code2)).all() and int(n_ac3) == int(n_ac), (
        "confirm scan diverges from plain scan on identical batches"
    )

    # (14) fitness acceptance objective == the GA's eval_pop fitness ingredients
    fm, fa = did_fitness(net, acts[-1], y)
    rm, ra = net.eval_pop(code[None], wa[None], wb[None], Xp, y)
    assert abs(float(fm) - float(rm[0])) < 1e-5 and abs(float(fa) - float(ra[0])) < 1e-6, (
        "did_fitness disagrees with eval_pop"
    )

    # (15) counterfactual rewire bins: Ca[g, k] must equal did_layer_c on a net whose gate g is
    # actually rewired to candidate k (lambda at g's output does not depend on g's own inputs),
    # and the pooled move delta must equal the joint (rewire + best table) surrogate identity
    for li in range(len(widths)):
        lo, hi = net.offs[li], net.offs[li + 1]
        prev = Xp if li == 0 else acts[li - 1]
        Ca, Cb = did_rewire_c(net, li, lams[li], prev, wa, wb)
        Ccur = did_layer_c(net, li, lams[li], prev, wa, wb)
        rng15 = np.random.default_rng(li)
        for g_l in rng15.integers(0, hi - lo, 3):
            for k in range(K):
                wa2 = wa.at[lo + int(g_l)].set(k)
                Ck = did_layer_c(net, li, lams[li], prev, wa2, wb)
                assert np.allclose(np.asarray(Ca[g_l, k]), np.asarray(Ck[g_l]), atol=1e-5), (
                    "rewire Ca vs rewired did_layer_c"
                )
                wb2 = wb.at[lo + int(g_l)].set(k)
                Ck = did_layer_c(net, li, lams[li], prev, wa, wb2)
                assert np.allclose(np.asarray(Cb[g_l, k]), np.asarray(Ck[g_l]), atol=1e-5), (
                    "rewire Cb vs rewired did_layer_c"
                )
        # candidate == current wire must reproduce the current bins exactly
        w_ar = np.arange(hi - lo)
        assert np.allclose(
            np.asarray(Ca)[w_ar, np.asarray(wa[lo:hi])], np.asarray(Ccur), atol=1e-6
        ), "rewire Ca at current wire != current bins"
        # move delta identity: best-response delta + base correction == t*.C_k - t.C_cur
        pk, dk = did_best_response(Ca, code[lo:hi, None])
        gb = code_bits(code[lo:hi])
        base = np.asarray(((Ca - Ccur[:, None, :]) * gb[:, None, :]).sum(-1))
        tstar = np.asarray(code_bits(pk))
        lhs = np.asarray(dk) + base
        rhs = (tstar * np.asarray(Ca)).sum(-1) - (np.asarray(gb) * np.asarray(Ccur)).sum(
            -1, keepdims=True
        )
        assert np.allclose(lhs, rhs, atol=1e-5), "rewire move-delta identity"

    # (16) joint rewire bins: Cj[g, u, v] must equal did_layer_c on a net with BOTH ports of
    # gate g rewired to (u, v), and must reduce to the per-port bins when one side stays current
    for li in range(len(widths)):
        lo, hi = net.offs[li], net.offs[li + 1]
        prev = Xp if li == 0 else acts[li - 1]
        Cj = did_rewire_c_joint(net, li, lams[li], prev)
        Ca, Cb = did_rewire_c(net, li, lams[li], prev, wa, wb)
        rng16 = np.random.default_rng(100 + li)
        for g_l in rng16.integers(0, hi - lo, 2):
            for u, v in rng16.integers(0, K, (3, 2)):
                wa2 = wa.at[lo + int(g_l)].set(int(u))
                wb2 = wb.at[lo + int(g_l)].set(int(v))
                Ck = did_layer_c(net, li, lams[li], prev, wa2, wb2)
                assert np.allclose(np.asarray(Cj[g_l, u, v]), np.asarray(Ck[g_l]), atol=1e-5), (
                    "joint Cj vs doubly-rewired did_layer_c"
                )
        w_ar = np.arange(hi - lo)
        assert np.allclose(
            np.asarray(Cj)[w_ar, np.asarray(wa[lo:hi]), :], np.asarray(Cb), atol=1e-4
        ), "joint bins at current a != per-port Cb"
        assert np.allclose(
            np.asarray(Cj)[w_ar, :, np.asarray(wb[lo:hi])], np.asarray(Ca), atol=1e-4
        ), "joint bins at current b != per-port Ca"

    # (17) two-gate (motif) proposals through the rewire scan == the wiring-fixed trial scan
    rng17 = np.random.default_rng(17)
    gm = jnp.asarray(rng17.integers(0, net.n_gates, 24).astype(np.int32))
    gm2 = jnp.asarray(rng17.integers(0, net.n_gates, 24).astype(np.int32))
    tm = jnp.asarray(rng17.integers(0, 16, 24).astype(np.uint8))
    tm2 = jnp.asarray(rng17.integers(0, 16, 24).astype(np.uint8))
    vld17 = jnp.asarray(np.ones(24, bool))
    z17 = jnp.zeros(24, jnp.int32)
    p17 = jnp.full(24, -1, jnp.int32)
    sa17, sb17 = net.sources(wa, wb)
    cA, lA, nA, aA, accA = did_trial_scan(net, code, wa, wb, Xp, y, gm, tm, gm2, tm2, vld17, loss)
    cB, _, _, lB, nB, aB, accB = did_rewire_trial_scan(
        net, code, sa17, sb17, Xp, y, gm, tm, gm2, tm2, p17, z17, z17, vld17, loss
    )
    assert (np.asarray(cA) == np.asarray(cB)).all() and int(aA) == int(aB), (
        "two-gate rewire scan diverges from the wiring-fixed trial scan"
    )
    assert (np.asarray(accA) == np.asarray(accB)).all() and abs(float(lA) - float(lB)) < 1e-5, (
        "two-gate rewire scan accept mask / loss mismatch"
    )

    # (18) distillation: soft-target head == autodiff of the soft-CE, its lambda/C survive the
    # same multilinear ground truth as the label path, one-hot targets reproduce the label path
    # exactly, and the rewire scan's carried loss under targets matches a recomputed forward.
    rng18 = np.random.default_rng(18)
    tsoft = jax.nn.softmax(jnp.asarray(rng18.normal(size=(N, classes)), jnp.float32), 1)
    loss_s, lam_s = did_head(net, acts[-1], y, tsoft)

    def loss_bits_soft(b):
        z = b.reshape(N, classes, group).sum(-1) / group**0.5
        return (jax.scipy.special.logsumexp(z, 1) - (z * tsoft).sum(1)).mean()

    assert abs(float(loss_bits_soft(bf)) - float(loss_s)) < 1e-6, "soft head loss mismatch"
    gs = jax.grad(loss_bits_soft)(bf)
    assert np.allclose(np.asarray(gs), np.asarray(lam_s), atol=1e-6), "soft output influence"
    lams_s = did_influence(net, code, wa, wb, acts, Xp, lam_s)
    c_soft = [
        did_layer_c(net, li, lams_s[li], Xp if li == 0 else acts[li - 1], wa, wb)
        for li in range(len(widths))
    ]

    def ml_loss_soft(tb):
        prev = Xf
        for mli in range(len(widths)):
            msa, msb = did_resolve(net, wa, wb, net.offs[mli], net.offs[mli + 1])
            a_, b_ = prev[:, msa], prev[:, msb]
            t = tb[mli]
            prev = (
                t[:, 0] * (1 - a_) * (1 - b_)
                + t[:, 1] * (1 - a_) * b_
                + t[:, 2] * a_ * (1 - b_)
                + t[:, 3] * a_ * b_
            )
        z = prev.reshape(N, classes, group).sum(-1) / group**0.5
        return (jax.scipy.special.logsumexp(z, 1) - (z * tsoft).sum(1)).mean()

    g_mls = jax.grad(ml_loss_soft)(tb_layers)
    for sli in range(len(widths)):
        assert np.allclose(np.asarray(g_mls[sli]), np.asarray(c_soft[sli]), atol=1e-5), (
            f"soft C vs multilinear autodiff, layer {sli}"
        )
    l_oh, lam_oh = did_head(net, acts[-1], y, jax.nn.one_hot(y, classes))
    assert abs(float(l_oh) - float(loss)) < 1e-7, "one-hot targets != label path (loss)"
    assert np.allclose(np.asarray(lam_oh), np.asarray(lam_out), atol=1e-7), (
        "one-hot targets != label path (lambda)"
    )
    cS, _, _, lS, _, _, _ = did_rewire_trial_scan(
        net, code, sa17, sb17, Xp, y, gm, tm, gm2, tm2, p17, z17, z17, vld17, loss_s, targets=tsoft
    )
    l_re = did_head(net, did_forward_layers(net, cS, wa, wb, Xp)[-1], y, tsoft)[0]
    assert abs(float(l_re) - float(lS)) < 1e-5, "soft scan carried loss mismatch"

    # (19) population distill fitness == -did_head soft loss for a single-genome population,
    # so GA selection under --distill descends the identical objective DID accepts on
    f19 = pop_softce(net, tab, wa[None], wb[None], Xp, tsoft)
    assert abs(float(f19[0]) + float(loss_s)) < 1e-5, "pop soft-CE fitness != did_head soft loss"

    print(
        "did selftest ok (forward, influence, C bins, best response, motifs, trial scan, "
        "batch-scale invariance, ranking calibration, fixed-batch plateau, head curvature, "
        "order-2 best response, confirm scan, fitness objective, rewire bins, joint bins, "
        "two-gate rewire scan, soft-target distillation, population distill fitness)"
    )


def load_genome(path: str):
    d = np.load(path)
    return jnp.asarray(d["tab"]), jnp.asarray(d["wa"]), jnp.asarray(d["wb"])


def main(cfg: Config) -> None:
    if cfg.selftest:
        did_selftest()
        return
    Xtr, ytr, Xte, yte = load_data(cfg.dataset, cfg.thresholds)
    net = Net(Xtr.shape[1], cfg.widths, 10, codebook=cfg.wire_codebook)
    print(
        f"algo={cfg.algo} dataset={cfg.dataset} net {Xtr.shape[1]} -> {cfg.widths} "
        f"({net.n_gates} gates, codebook K={cfg.wire_codebook}) backend={jax.default_backend()}",
        flush=True,
    )
    r = Runner(net, Xtr, ytr, Xte, yte, cfg)
    t0 = time.time()
    r.run()
    train_seconds = time.time() - t0

    gate_evals = r.evals * cfg.batch * net.n_gates
    metrics = {
        "run_name": cfg.run_name or cfg.algo,
        "algo": cfg.algo,
        "dataset": cfg.dataset,
        "wire_codebook": cfg.wire_codebook,
        "test_acc": round(r.best[1], 4),
        "model_memory_bytes": net.model_memory_bytes(),
        "n_gates": net.n_gates,
        "pop": cfg.pop,
        "gens": cfg.gens,
        "evaluations": r.evals,  # budget-matched across algorithms
        "gate_evaluations": gate_evals,
        "train_flops": gate_evals,
        "train_seconds": round(train_seconds, 1),
        "gpu_count": 1,
        "init_genome": cfg.init_genome,
    }
    print("METRICS " + json.dumps(metrics), flush=True)
    if cfg.metrics_out:
        Path(cfg.metrics_out).write_text(json.dumps(metrics, indent=2))
    if cfg.save_genome and r.best[0] is not None:
        bt, bwa, bwb = r.best[0]
        Path(cfg.save_genome).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cfg.save_genome, tab=np.asarray(bt), wa=np.asarray(bwa), wb=np.asarray(bwb))
        print(f"saved best genome -> {cfg.save_genome}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Config))
