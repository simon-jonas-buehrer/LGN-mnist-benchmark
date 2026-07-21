"""Why are two thirds of the gates dead? Decompose the dead fraction by cause, per layer.

The sensitivity backward pass found ~65% of gates have zero influence. Before "fixing" that, this
script attributes every dead gate to one of three causes, because each implies a different fix:

  unread    — no consumer wire points at the gate in the REALIZED wiring. An i.i.d.-wired funnel
              guarantees a lot of this: layer k+1 has only 2*w_{k+1} wire slots to read w_k sources,
              so at most 2*w_{k+1}/w_k of a layer can be live no matter what evolution does.
  masked    — read by at least one consumer, but every consumer's LUT is insensitive to that port
              (e.g. the consumer learned copy-A and reads this gate on port B).
  dead-path — read AND locally sensitive, but every sensitive consumer is itself dead, so the
              influence never reaches an output bit.

Also reports the architecture's hard cap on live gates per layer, and the codebook's reachability
(a source no candidate ever offers can never be revived by evolution).

Run:
    uv run --no-project --with jax --with tyro python dead_gate_autopsy.py
"""

from __future__ import annotations

import jax
import numpy as np
import tyro
from ga_bits_wiring_mnist import Net, load_data, pack
from gate_importance_mnist import Config, sensitivity, train


def autopsy(net: Net, tab1, wa1, wb1, X):
    """Per-gate cause-of-death labels for one genome."""
    sens = np.asarray(sensitivity(net, tab1, wa1, wb1, X))
    dead = sens == 0

    code = np.asarray(net.codes(tab1[None])[0])
    sa, sb = net.sources(wa1, wb1)
    sa, sb = np.asarray(sa), np.asarray(sb)
    c0, c1, c2, c3 = (code & 1), ((code >> 1) & 1), ((code >> 2) & 1), ((code >> 3) & 1)
    # port-level STATIC sensitivity of each consumer gate (any input combination):
    # gate can respond to port a iff f(0,b) != f(1,b) for SOME b, i.e. f00!=f10 or f01!=f11
    sens_a = ((c0 ^ c2) | (c1 ^ c3)).astype(bool)  # (n_gates,)
    sens_b = ((c0 ^ c1) | (c2 ^ c3)).astype(bool)

    L = len(net.widths)
    unread = np.zeros(net.n_gates, bool)
    masked = np.zeros(net.n_gates, bool)
    for k in range(L - 1):  # output layer is read by the popcount head, never unread
        lo, hi = net.offs[k + 1], net.offs[k + 2]  # consumer layer
        w = net.widths[k]
        read = np.zeros(w, bool)
        live_read = np.zeros(w, bool)  # read through a SENSITIVE port
        np.logical_or.at(read, sa[lo:hi], True)
        np.logical_or.at(read, sb[lo:hi], True)
        np.logical_or.at(live_read, sa[lo:hi], sens_a[lo:hi])
        np.logical_or.at(live_read, sb[lo:hi], sens_b[lo:hi])
        seg = slice(net.offs[k], net.offs[k + 1])
        unread[seg] = ~read
        masked[seg] = read & ~live_read
    deadpath = dead & ~unread & ~masked  # sensitive wire exists, but influence never reaches out
    return sens, dead, unread, masked, deadpath


def main(cfg: Config) -> None:
    Xtr, ytr, Xte, yte = load_data(cfg.dataset, cfg.thresholds)
    net = Net(Xtr.shape[1], cfg.widths, 10, codebook=cfg.wire_codebook, dag=cfg.dag)
    print(f"training {cfg.gens} gens to get a real circuit...", flush=True)
    tab, wa, wb = train(net, Xtr, ytr, cfg)
    _, acc = net.eval_pop(net.codes(tab[:1]), wa[:1], wb[:1], pack(Xte), yte)
    print(f"trained circuit: TEST {float(acc[0]):.4f}\n", flush=True)

    kp = jax.random.PRNGKey(cfg.seed + 7)
    pi = jax.random.randint(kp, (cfg.probe,), 0, Xtr.shape[0])
    sens, dead, unread, masked, deadpath = autopsy(net, tab[0], wa[0], wb[0], Xtr[pi])

    print(
        f"total dead: {dead.mean():.1%}  =  unread {unread.mean():.1%} "
        f"+ masked {(dead & masked).mean():.1%} + dead-path {deadpath.mean():.1%}  "
        f"(+ head-layer effects)\n"
    )
    print(
        f"{'layer':>6} {'width':>6} {'cap(live)':>10} {'dead':>7} {'unread':>7} {'masked':>7} {'deadpath':>8}"
    )
    for k in range(len(net.widths)):
        seg = slice(net.offs[k], net.offs[k + 1])
        w = net.widths[k]
        cap = min(1.0, 2 * net.widths[k + 1] / w) if k < len(net.widths) - 1 else 1.0
        print(
            f"{k:>6} {w:>6} {cap:>10.1%} {dead[seg].mean():>7.1%} {unread[seg].mean():>7.1%} "
            f"{(dead & masked)[seg].mean():>7.1%} {deadpath[seg].mean():>8.1%}"
        )

    # codebook reachability: which sources appear in NO gate's candidate list (per layer)?
    print("\ncodebook reachability (sources of each layer offered by >=1 candidate):")
    ca, cb = np.asarray(net.cand_a), np.asarray(net.cand_b)  # (K, n_gates)
    prev_widths = [net.n_in, *net.widths[:-1]]
    for k in range(len(net.widths)):
        lo, hi = net.offs[k], net.offs[k + 1]
        offered = np.zeros(prev_widths[k], bool)
        np.logical_or.at(offered, ca[:, lo:hi].ravel(), True)
        np.logical_or.at(offered, cb[:, lo:hi].ravel(), True)
        print(f"  layer {k} sources ({prev_widths[k]}): reachable {offered.mean():.1%}")

    # how much of the dead mass could ANY wiring fix? (the architecture cap)
    cap_live = (
        sum(min(w, 2 * net.widths[k + 1]) for k, w in enumerate(net.widths[:-1])) + net.widths[-1]
    )
    print(
        f"\narchitecture cap: at most {cap_live}/{net.n_gates} gates "
        f"({cap_live / net.n_gates:.1%}) can be live under ANY wiring of this funnel"
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
