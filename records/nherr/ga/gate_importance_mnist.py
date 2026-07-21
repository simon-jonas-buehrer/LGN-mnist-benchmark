"""Which gates matter? Three measures of per-gate importance, cross-validated against each other.

Crossover is the GA's whole advantage (removing it costs 6.5 points), so the natural next lever is to
stop treating every gate as equally worth inheriting. That needs a trustworthy notion of *gate
importance*. This script implements three, cheap to expensive, and checks they agree:

  1. sensitivity  — a BACKWARD PASS over the circuit. A 2-input LUT tells you exactly when it is
                    sensitive to each input (f(0,b) != f(1,b)), so influence propagates backwards like
                    a Boolean version of backprop. importance(g) = mean number of output bits that flip
                    when g flips. Costs about one forward pass. Also gives the dead-gate fraction free.
  2. conservation — across the top elites, how locked-down is this gate? A gate whose (table, wiring)
                    is identical in every elite has been pinned by selection; one that varies freely is
                    probably neutral. Nearly free — a statistic over the final population.
  3. knockout     — ground truth. Force the gate to a constant and measure the drop in fitness.
                    Expensive (one evaluation per gate), so it is run on a random sample.

(1) and (2) are the cheap proxies we would actually use inside crossover; (3) is the expensive truth we
check them against. If they do not correlate with (3), they are not worth using.

Run:
    uv run --no-project --with jax --with tyro python gate_importance_mnist.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import tyro
from evo_algos_mnist import crossover, mutate
from ga_bits_wiring_mnist import Net, load_data, pack


@dataclass
class Config:
    dataset: str = "mnist"
    widths: list[int] = field(default_factory=lambda: [3072, 1024, 500])
    thresholds: list[int] = field(default_factory=lambda: [32, 64, 96, 128, 160, 192, 224])
    wire_codebook: int = 8
    dag: bool = False  # True = a gate may read ANY earlier signal, not just the previous layer
    pop: int = 256
    gens: int = 3000  # enough to get a genuinely trained circuit to analyse
    batch: int = 8000
    mut: float = 0.005
    mut_end: float = 0.0004
    wire_mut: float = 0.003
    wire_mut_end: float = 0.0002
    acc_weight: float = 100.0
    tsize: int = 5
    elite: int = 4
    n_elite: int = 32  # elites used for the conservation statistic
    probe: int = 2048  # samples the sensitivity backward pass is averaged over
    n_knockout: int = 400  # gates sampled for the (expensive) ground-truth ablation
    seed: int = 0


def train(net: Net, Xtr, ytr, cfg: Config):
    """A short GA run — we need a genuinely trained circuit, not a random one, to analyse."""
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


# --------------------------------------------------------------------------------------
# 1. sensitivity: a Boolean backward pass
# --------------------------------------------------------------------------------------
def sensitivity(net: Net, tab1, wa1, wb1, X):
    """importance(g) = mean # of output bits that flip when gate g's output flips.

    A 2-input LUT with truth table (f00,f01,f10,f11) is sensitive to input `a` iff flipping `a` flips
    the output *given the current value of b* — i.e. f(0,b) != f(1,b). That is a per-sample bit we can
    read straight off the table, so influence propagates backwards through the circuit exactly.
    """
    code = net.codes(tab1[None])[0]  # (n_gates,)
    sa_all, sb_all = net.sources(wa1, wb1)  # actual source indices
    c0, c1 = (code & 1), ((code >> 1) & 1)
    c2, c3 = ((code >> 2) & 1), ((code >> 3) & 1)

    # ---- forward (unpacked), recording each gate's two input bits ----
    prev = X.T.astype(jnp.int32)  # (n_in, N)
    A, B, SA, SB = [], [], [], []
    for k in range(len(net.widths)):
        lo, hi = net.offs[k], net.offs[k + 1]
        sa, sb = sa_all[lo:hi], sb_all[lo:hi]
        a, b = prev[sa], prev[sb]  # (w, N)
        k0, k1 = c0[lo:hi, None], c1[lo:hi, None]
        k2, k3 = c2[lo:hi, None], c3[lo:hi, None]
        out = (1 - a) * (1 - b) * k0 + (1 - a) * b * k1 + a * (1 - b) * k2 + a * b * k3
        # sensitive to a (given b) / to b (given a) — read straight off the truth table
        SA.append(jnp.where(b == 0, k0 ^ k2, k1 ^ k3))
        SB.append(jnp.where(a == 0, k0 ^ k1, k2 ^ k3))
        A.append(sa)
        B.append(sb)
        prev = out

    # ---- backward: how many output bits does a flip of this gate reach? ----
    L = len(net.widths)
    infl = [None] * L
    infl[L - 1] = jnp.ones((net.widths[-1], X.shape[0]), jnp.int32)  # an output bit IS the output
    for k in range(L - 2, -1, -1):
        acc = jnp.zeros((net.widths[k], X.shape[0]), jnp.int32)
        acc = acc.at[A[k + 1]].add(SA[k + 1] * infl[k + 1])  # via consumers' input-A port
        acc = acc.at[B[k + 1]].add(SB[k + 1] * infl[k + 1])  # ... and input-B port
        infl[k] = acc
    return jnp.concatenate([i.mean(1) for i in infl])  # (n_gates,)


# --------------------------------------------------------------------------------------
# 2. conservation across elites
# --------------------------------------------------------------------------------------
def conservation(net: Net, tab, wa, wb, n_elite: int):
    """Fraction of the top elites that share the modal (truth table, wire A, wire B) for each gate."""
    code = np.asarray(net.codes(tab[:n_elite]))  # (E, n_gates)
    a, b = np.asarray(wa[:n_elite]), np.asarray(wb[:n_elite])
    K = net.codebook or 1
    gene = code.astype(np.int64) * K * K + a.astype(np.int64) * K + b.astype(np.int64)
    E = gene.shape[0]
    out = np.zeros(gene.shape[1], np.float32)
    for g in range(gene.shape[1]):  # per gate: modal frequency among the elites
        out[g] = np.bincount(gene[:, g]).max() / E
    return jnp.asarray(out)


# --------------------------------------------------------------------------------------
# 3. knockout ablation (ground truth)
# --------------------------------------------------------------------------------------
def knockout(net: Net, tab1, wa1, wb1, gates, Xp, y, cfg: Config):
    """Force each sampled gate to the constant-0 function; measure the fitness drop. Batched."""

    def fit_of(t, a, b):
        margin, acc = net.eval_pop(net.codes(t), a, b, Xp, y)
        return margin + cfg.acc_weight * acc

    base = float(fit_of(tab1[None], wa1[None], wb1[None])[0])
    T = jnp.repeat(tab1[None], len(gates), 0)  # one copy per sampled gate
    idx = jnp.asarray(gates)
    # a gate's 4 table bits live at [4g : 4g+4]; zeroing them = the constant-0 function
    for j in range(4):
        T = T.at[jnp.arange(len(gates)), idx * 4 + j].set(0)
    A = jnp.repeat(wa1[None], len(gates), 0)
    B = jnp.repeat(wb1[None], len(gates), 0)
    return base - fit_of(T, A, B)  # drop in fitness = how much the gate was worth


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx @ ry) / (np.linalg.norm(rx) * np.linalg.norm(ry) + 1e-12))


def main(cfg: Config) -> None:
    Xtr, ytr, Xte, yte = load_data(cfg.dataset, cfg.thresholds)
    net = Net(Xtr.shape[1], cfg.widths, 10, codebook=cfg.wire_codebook, dag=cfg.dag)
    print(
        f"training {cfg.gens} gens (pop {cfg.pop}) to get a real circuit to analyse...", flush=True
    )
    tab, wa, wb = train(net, Xtr, ytr, cfg)
    _, acc = net.eval_pop(net.codes(tab[:1]), wa[:1], wb[:1], pack(Xte), yte)
    print(f"trained circuit: TEST {float(acc[0]):.4f}\n", flush=True)

    key = jax.random.PRNGKey(cfg.seed + 7)
    kp, kg, kb = jax.random.split(key, 3)
    pi = jax.random.randint(kp, (cfg.probe,), 0, Xtr.shape[0])

    print("[1] sensitivity backward pass ...", flush=True)
    sens = np.asarray(sensitivity(net, tab[0], wa[0], wb[0], Xtr[pi]))
    dead = float((sens == 0).mean())

    print("[2] conservation across elites ...", flush=True)
    cons = np.asarray(conservation(net, tab, wa, wb, cfg.n_elite))

    print(f"[3] knockout ablation on {cfg.n_knockout} sampled gates ...", flush=True)
    gates = np.asarray(jax.random.choice(kg, net.n_gates, (cfg.n_knockout,), replace=False))
    bi = jax.random.randint(kb, (cfg.batch,), 0, Xtr.shape[0])
    Xp, y = pack(Xtr[bi]), ytr[bi]
    ko = np.asarray(knockout(net, tab[0], wa[0], wb[0], gates, Xp, y, cfg))

    trunk = gates < net.offs[-2]  # gates outside the final (output) layer
    print(f"\ndead gates (zero influence): {dead:.1%}")
    print(f"knockout drop: mean {ko.mean():+.3f}  max {ko.max():+.3f}  min {ko.min():+.3f}")
    print("\nconsistency (Spearman rank correlation vs the ground-truth knockout):")
    print(f"  sensitivity  vs knockout : {spearman(sens[gates], ko):+.3f}   (all gates)")
    print(
        f"  sensitivity  vs knockout : {spearman(sens[gates][trunk], ko[trunk]):+.3f}   (trunk only)"
    )
    print(f"  conservation vs knockout : {spearman(cons[gates], ko):+.3f}   (all gates)")
    print(
        f"  conservation vs knockout : {spearman(cons[gates][trunk], ko[trunk]):+.3f}   (trunk only)"
    )
    print(f"  sensitivity  vs conservation : {spearman(sens, cons):+.3f}   (all gates)")

    # the sharpest check: gates the backward pass calls DEAD should be worthless to knock out
    dz = sens[gates] == 0
    if dz.any() and (~dz).any():
        print(
            f"\ndead-gate check: knockout drop is {ko[dz].mean():+.4f} for gates the backward pass "
            f"calls dead, vs {ko[~dz].mean():+.4f} for live gates"
        )


if __name__ == "__main__":
    main(tyro.cli(Config))
