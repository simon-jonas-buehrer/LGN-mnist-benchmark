"""The GA: tournament selection, gate-wise crossover, annealed mutation, elitism.

One generation is one fresh minibatch. Parents are picked by tournament on `margin + acc_weight *
accuracy`; crossover takes each gate -- its truth table AND both its wires -- wholesale from one
parent or the other, so a gate that co-adapted its function to its inputs keeps that pairing;
mutation flips table bits and rewires ports at rates annealed geometrically over the run; the top
`elite` genomes survive untouched.

`train` returns the final population sorted best-first, so `[0]` is the genome the record ships.

Carved verbatim out of `evo_algos_mnist.py` (mutate, crossover) and `gate_importance_mnist.py`
(Config, train) in the record this came from. The other seven algorithms it compared against, and
the importance measures, are not part of the measured path and were left behind.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from lutnet import Net, pack


@dataclass
class Config:
    """One GA run. Total network evaluations = pop * gens."""

    widths: list[int] = field(default_factory=lambda: [3072, 1024, 500])
    thresholds: list[int] = field(default_factory=lambda: [32, 64, 96, 128, 160, 192, 224])
    wire_codebook: int = 8  # K candidate wirings per gate (the cheap-to-store scheme)
    pop: int = 256
    gens: int = 3000
    batch: int = 8000  # samples per generation; selection noise is what this buys down
    mut: float = 0.005  # table bit-flip rate at gen 0, annealed geometrically to mut_end
    mut_end: float = 0.0004
    wire_mut: float = 0.003  # per-gate rewire rate at gen 0, annealed to wire_mut_end
    wire_mut_end: float = 0.0002
    acc_weight: float = 100.0  # fitness = margin + acc_weight * batch_accuracy
    tsize: int = 5  # tournament size
    elite: int = 4  # genomes carried over untouched
    seed: int = 0


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


def train(net: Net, Xtr, ytr, cfg: Config):
    """Run the GA for cfg.gens generations; returns the population sorted best-first."""
    key = jax.random.PRNGKey(cfg.seed)
    key, ki = jax.random.split(key)
    tab, wa, wb = net.init_pop(ki, cfg.pop)
    ar = jnp.arange(cfg.pop)
    for gen in range(cfg.gens):
        f = gen / max(cfg.gens - 1, 1)
        m = cfg.mut * (cfg.mut_end / cfg.mut) ** f
        wm = cfg.wire_mut * (cfg.wire_mut_end / cfg.wire_mut) ** f
        key, kb, kt, kc, km = jax.random.split(key, 5)
        idx = jax.random.randint(kb, (cfg.batch,), 0, Xtr.shape[0])
        margin, acc = net.eval_pop(net.codes(tab), wa, wb, pack(Xtr[idx]), ytr[idx])
        fit = margin + cfg.acc_weight * acc
        order = jnp.argsort(-fit)
        tab, wa, wb = tab[order], wa[order], wb[order]
        if gen == cfg.gens - 1:
            break
        cand = jax.random.randint(kt, (cfg.pop, cfg.tsize), 0, cfg.pop)
        win = cand[ar, fit[order][cand].argmax(1)]
        pt, pa, pb = tab[win], wa[win], wb[win]
        ct, ca, cb = crossover(kc, net, pt, pa, pb)
        ct, ca, cb = mutate(km, net, ct, ca, cb, m, wm)
        tab = ct.at[: cfg.elite].set(pt[: cfg.elite])
        wa = ca.at[: cfg.elite].set(pa[: cfg.elite])
        wb = cb.at[: cfg.elite].set(pb[: cfg.elite])
    return tab, wa, wb  # sorted best-first
