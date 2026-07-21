"""Evolutionary algorithms for a binary LUT network — Aging Evolution, MAP-Elites, NSLC vs the GA.

Implements the comparison recommended in `documents/evolutionary_algorithms_for_nn_design.md`: hold the
**representation, mutation operators, fitness, and evaluation budget fixed**, and change only the
evolutionary algorithm. The circuit, genome (truth tables + wiring), and mutation are imported straight
from `ga_bits_wiring_mnist.py`, so the only variable is the search algorithm.

Algorithms (`--algo`):
  ga         the incumbent: tournament selection + gate-uniform crossover + elitism.
  aging      Aging (regularized) Evolution — tournament -> mutate -> append child, evict the OLDEST.
             Removing old members stops one early winner from owning the population forever. The
             document's recommended *baseline*; mutation-only.
  mapelites  MAP-Elites — a 2D archive of niches. Keep the best genome per niche and breed from the
             archive, so behaviourally different circuits are preserved instead of collapsing to one.
             Descriptors: (prediction entropy, output activation rate) — see `descriptors`.
  nslc       Novelty Search with Local Competition — reward behavioural novelty (distance to k nearest
             behaviours seen) AND beating your behavioural neighbours, so promising-but-immature
             circuits survive as stepping stones.
  eda        Univariate EDA (UMDA/PBIL) — carry a *distribution* over genomes (a Bernoulli per table
             bit, a Categorical(K) per wire), sample it, and refit the marginals to the elites.
  snes       Separable Natural Evolution Strategies — keep a Gaussian over continuous latents, and turn
             the population into an *estimated gradient* of expected fitness, then step along it.

`eda` and `snes` replace the mutation/crossover operators outright (they carry a distribution rather
than a population). That is deliberate: in ~28k discrete dimensions, blind mutation + crossover
propagate far less information per evaluation than marginals or an estimated gradient do — which is the
GA's suspected weakness here. They are still compared on the same representation, fitness and budget.

Every algorithm spends the SAME number of network evaluations: `pop * gens`.

Run:
    uv run --no-project --with jax --with tyro python evo_algos_mnist.py --algo mapelites
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tyro
from ga_bits_wiring_mnist import Net, load_data, pack


@dataclass
class Config:
    """Shared settings — identical across algorithms so only the search differs."""

    algo: str = "ga"  # ga | aging | mapelites | nslc
    dataset: str = "mnist"  # mnist | cifar10
    widths: list[int] = field(default_factory=lambda: [3072, 1024, 500])
    thresholds: list[int] = field(default_factory=lambda: [32, 64, 96, 128, 160, 192, 224])
    wire_codebook: int = 8  # K candidate wirings per gate (the cheap-to-store scheme)
    dag: bool = False  # True = a gate may read ANY earlier signal, not just the previous layer
    #   (graph encoding). Size-neutral under the codebook — only the candidate pool widens.
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
    sterilize_every: int = 0  # ga only: every N gens wipe DEAD gates' tables (0 = off) — the
    #   scratch-space test: if dead gates are a useful neutral reservoir, wiping them should hurt
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
    seed: int = 0
    log_every: int = 2000
    run_name: str = ""
    metrics_out: str = ""


# --------------------------------------------------------------------------------------
# Shared genome ops — IDENTICAL for every algorithm (the document's fixed-operator rule)
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


def descriptors(net: Net, logits):
    """MAP-Elites niche coordinates, both in [0,1] — BEHAVIOURAL, so they actually spread.

    An earlier attempt used genome statistics (dead-gate fraction, wiring span). Both concentrate
    tightly around their random-init values (0.125 and 0.33), so every genome landed in one cell and
    the archive degenerated to a hill-climber — the descriptor-choice failure the source note warns of.

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

    def split(self, n):
        self.key, *ks = jax.random.split(self.key, n + 1)
        return ks

    def fitness(self, tab, wa, wb, kb):
        """One evaluation of a batch of genomes on a fresh minibatch (the shared fitness)."""
        idx = jax.random.randint(kb, (self.cfg.batch,), 0, self.Xtr.shape[0])
        margin, acc = self.net.eval_pop(
            self.net.codes(tab), wa, wb, pack(self.Xtr[idx]), self.ytr[idx]
        )
        self.evals += tab.shape[0]
        return margin + self.cfg.acc_weight * acc

    def test_best(self, tab, wa, wb):
        _, acc = self.net.eval_pop(
            self.net.codes(tab[None]), wa[None], wb[None], self.Xte_p, self.yte
        )
        v = float(acc[0])
        if v > self.best[1]:
            self.best = ((tab, wa, wb), v)
        return v

    def rates(self, gen):
        f = gen / max(self.cfg.gens - 1, 1)
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
        ar = jnp.arange(c.pop)
        imp = None  # per-gate importance in [0,1] (refreshed every bias_every gens)
        for gen in range(c.gens):
            mut, wmut = self.rates(gen)
            kb, kt, kc, km = self.split(4)
            if c.sterilize_every and gen and gen % c.sterilize_every == 0:
                # scratch-space test: dead gates cost nothing to wipe (static liveness => the output
                # is bit-identical), so any accuracy loss must come from the evolved content they
                # were holding — i.e. they were a useful neutral reservoir, not junk.
                dead = static_dead(net, tab, wa, wb)
                (ks,) = self.split(1)
                fresh = (
                    jax.random.bernoulli(ks, 0.5, tab.shape).astype(tab.dtype)
                    if c.sterilize_mode == "rand"
                    else jnp.zeros_like(tab)
                )
                tab = jnp.where(jnp.repeat(dead, 4, axis=1), fresh, tab)
            fit = self.fitness(tab, wa, wb, kb)
            order = jnp.argsort(-fit)
            tab, wa, wb, fit = tab[order], wa[order], wb[order], fit[order]
            self.cur = (tab[0], wa[0], wb[0])
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
            ct, ca, cb = mutate(km, net, ct, ca, cb, mut, wmut)
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
        }[self.cfg.algo]()


def main(cfg: Config) -> None:
    Xtr, ytr, Xte, yte = load_data(cfg.dataset, cfg.thresholds)
    net = Net(Xtr.shape[1], cfg.widths, 10, codebook=cfg.wire_codebook, dag=cfg.dag)
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
    }
    print("METRICS " + json.dumps(metrics), flush=True)
    if cfg.metrics_out:
        Path(cfg.metrics_out).write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Config))
