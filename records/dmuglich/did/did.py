"""DID -- discrete influence descent: one network, improved by ranked discrete moves.

Each sweep draws one fresh batch and does three things:

  1. Linearise. A closed-form output influence at the popcount head is backpropagated through the
     circuit as a SIGNED Boolean sensitivity -- a 2-input LUT tells you exactly when it is
     sensitive to each input -- giving per-gate, per-pattern C coefficients: what the loss would
     do if this gate's output flipped on this input pattern.
  2. Propose. C turns into candidate moves, all ranked in ONE global pool by surrogate delta:
     single truth-table rows; parent-child motifs (roll each of the 16 parent tables through
     cached activations, pair with the child's closed-form best response, score as one two-gate
     move); and codebook rewires (point a port at another of the K candidates, scored jointly
     with the best-response table -- `did_joint` also scores all K^2 two-port pairs).
  3. Accept. The top `did_props` proposals are tried ONE AT A TIME by exact forward pass and kept
     only on a measured loss drop. The linearisation proposes; it never decides. `did_dedup` keeps
     one proposal per gate, because ~85% of a raw top-512 targets a gate a better-ranked entry
     already claimed and would spend its trial on a stale genome.

Every trial is charged against the same evaluation budget (`pop * gens`) a population search gets,
which is what makes this comparable to `records/nherr/ga` rather than merely similar.

Carved verbatim out of `evo_algos_mnist.py` in the record this came from. That file held eight
budget-matched algorithms behind one `Runner`; the seven that are not DID, and the DID unit-test
suite, are not part of the measured path and were left behind.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from lutnet import Net, pack


@dataclass
class Config:
    """One DID run. The budget is `pop * gens` network evaluations, however they are spent."""

    widths: list[int] = field(default_factory=lambda: [3072, 1024, 500])
    thresholds: list[int] = field(default_factory=lambda: [32, 64, 96, 128, 160, 192, 224])
    wire_codebook: int = 8  # K candidate wirings per gate (the cheap-to-store scheme)
    pop: int = 512  # budget unit: total evaluations = pop * gens, matched across searches
    gens: int = 20000
    batch: int = 8000  # samples per evaluation
    acc_weight: float = 100.0  # fitness = margin + acc_weight * batch_accuracy (did_accept_fit)
    mut: float = 0.005  # mutation rates: unused by DID itself, read by the shared anneal schedule
    mut_end: float = 0.0004
    wire_mut: float = 0.003
    wire_mut_end: float = 0.0002
    did_props: int = 512  # exact-acceptance trials per sweep, taken in global priority order
    did_rewire: bool = False  # add codebook rewire proposals (per-port counterfactual C bins,
    #   joint best-response table) to the singleton pool
    did_joint: bool = False  # with did_rewire: also score all K^2 joint (u, v) candidate pairs
    #   per gate -- two-port moves the port-separable bins cannot see
    did_dedup: bool = False  # trial only the best-ranked proposal per written gate
    did_parent_child: bool = False  # add counterfactual parent-child (2-gate) motif proposals
    did_order2: bool = False  # curvature-damped (diagonal-GGN) proposal scoring
    did_ema: float = 0.0  # EMA factor over the C bins across sweeps (0 = single batch)
    did_confirm: bool = False  # accepts must also improve an independent batch (2x cost)
    did_accept_fit: bool = False  # accept on margin + acc_weight * acc, not CE
    did_accept_t0: float = 0.0  # Metropolis acceptance temperature, annealed to t0/1000 (0 = exact)
    batch_end: int = 0  # anneal the effective batch to this size over the second half of the
    #   budget; charged pro-rata, so the run still ends when the budget does
    elite_reval: int = 0  # population-search knob; kept because the anneal schedule reads it
    init_genome: str = ""  # start from a saved genome (.npz: tab, wa, wb) instead of a fresh one
    seed: int = 0


def load_genome(path: str):
    d = np.load(path)
    return jnp.asarray(d["tab"]), jnp.asarray(d["wa"]), jnp.asarray(d["wb"])




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


class Runner:
    """One DID run. `Xte`/`yte` are the HELD-OUT set the run keeps its best genome by -- give it
    validation, never test."""

    def __init__(self, net: Net, Xtr, ytr, Xte, yte, cfg: Config):
        self.net, self.cfg = net, cfg
        self.Xtr, self.ytr = Xtr, ytr
        self.Xte_p, self.yte = pack(Xte), yte
        self.key = jax.random.PRNGKey(cfg.seed)
        self.best = (None, 0.0)  # (genome, held-out acc)
        self.evals = 0
        # the upstream file also carried teacher-distilled targets here; DID runs on hard labels,
        # so the target rows stay one-hot and `tgt_all` stays None all the way through run_did
        self.tgt_all = None

    def split(self, n):
        self.key, *ks = jax.random.split(self.key, n + 1)
        return ks

    def fitness(self, tab, wa, wb, kb, m: int = 1):
        """One evaluation of a batch of genomes on a fresh minibatch (the shared fitness).

        With m > 1 the estimate averages m INDEPENDENT fresh base-size batches — samples are iid
        draws with replacement, so this is statistically identical to one m-times-larger batch,
        but reuses the same compiled kernel at the same peak memory. Charged m evaluations per
        genome."""
        codes = self.net.codes(tab)
        acc_f = 0.0
        for k in jax.random.split(kb, m):
            idx = jax.random.randint(k, (self.cfg.batch,), 0, self.Xtr.shape[0])
            margin, acc = self.net.eval_pop(codes, wa, wb, pack(self.Xtr[idx]), self.ytr[idx])
            acc_f = acc_f + margin + self.cfg.acc_weight * acc
        self.evals += tab.shape[0] * m
        return acc_f / m

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
            t_prop = t_acc = tgt
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

    def cur_best(self):
        return self.cur

    def run(self):
        return self.run_did()
