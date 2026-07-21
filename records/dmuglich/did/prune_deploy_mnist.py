"""Deploy-time dead-gate pruning: stop storing the ~2/3 of the circuit that provably does nothing.

The autopsy showed most gates are dead — unread, or read only through LUT ports the consumer is
insensitive to. Those gates can be dropped from the deployed artifact with ZERO behaviour change.
This script makes that exact and measures the size win:

  1. Train the GA (codebook wiring) as usual.
  2. Compute STATIC liveness — data-independent, so pruning is bit-exact for ALL inputs, not just a
     probe set: a gate is live iff some chain of statically-sensitive ports connects it to an output
     bit. (Static port sensitivity: consumer responds to port a iff f(0,b) != f(1,b) for SOME b.)
  3. Verify: zero out every statically-dead gate and check predictions are IDENTICAL on the full
     test set.
  4. Report the pruned artifact size: a live-mask bitmap (n_gates bits — learned structure, must be
     stored) + tables (4 b) and codebook choices (2*log2 K b) for LIVE gates only.

Run:
    uv run --no-project --with jax --with tyro python prune_deploy_mnist.py --gens 20000 --pop 512
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import tyro
from ga_bits_wiring_mnist import Net, load_data, pack
from gate_importance_mnist import Config, train


@dataclass
class PruneConfig(Config):
    genome: str = ""  # prune this saved genome (.npz: tab, wa, wb) instead of training one


def static_live(net: Net, tab1, wa1, wb1) -> np.ndarray:
    """Data-independent liveness: can a flip of this gate EVER reach an output bit?

    live[output layer] = True (the popcount head reads every output bit). Going backwards, gate g is
    live iff some LIVE consumer reads it through a port its LUT is statically sensitive to.
    """
    code = np.asarray(net.codes(tab1[None])[0])
    sa, sb = net.sources(wa1, wb1)
    sa, sb = np.asarray(sa), np.asarray(sb)
    c0, c1, c2, c3 = (code & 1), ((code >> 1) & 1), ((code >> 2) & 1), ((code >> 3) & 1)
    sens_a = ((c0 ^ c2) | (c1 ^ c3)).astype(bool)  # responds to port a for some b
    sens_b = ((c0 ^ c1) | (c2 ^ c3)).astype(bool)

    live = np.zeros(net.n_gates, bool)
    live[net.offs[-2] :] = True  # output layer
    for k in range(len(net.widths) - 2, -1, -1):
        lo, hi = net.offs[k + 1], net.offs[k + 2]  # consumer layer
        cons_live = live[lo:hi]
        seg_live = np.zeros(net.widths[k], bool)
        np.logical_or.at(seg_live, sa[lo:hi], sens_a[lo:hi] & cons_live)
        np.logical_or.at(seg_live, sb[lo:hi], sens_b[lo:hi] & cons_live)
        live[net.offs[k] : net.offs[k + 1]] = seg_live
    return live


def pruned_bytes(net: Net, n_live: int) -> int:
    """Artifact bytes after pruning: live-mask bitmap + tables & wire choices of live gates only."""
    K = net.codebook or 1
    wire_bits = 2 * max(1, math.ceil(math.log2(K))) if K > 1 else 0
    return math.ceil((net.n_gates + n_live * (4 + wire_bits)) / 8)


def main(cfg: PruneConfig) -> None:
    Xtr, ytr, Xte, yte = load_data(cfg.dataset, cfg.thresholds)
    net = Net(Xtr.shape[1], cfg.widths, 10, codebook=cfg.wire_codebook)
    if cfg.genome:
        z = np.load(cfg.genome)
        t1, a1, b1 = jnp.asarray(z["tab"]), jnp.asarray(z["wa"]), jnp.asarray(z["wb"])
    else:
        print(f"training {cfg.gens} gens (pop {cfg.pop})...", flush=True)
        tab, wa, wb = train(net, Xtr, ytr, cfg)
        t1, a1, b1 = tab[0], wa[0], wb[0]
    Xte_p = pack(Xte)

    live = static_live(net, t1, a1, b1)
    n_live = int(live.sum())
    print(f"\nstatic liveness: {n_live}/{net.n_gates} live ({live.mean():.1%})", flush=True)

    # verify bit-exactness on the FULL test set: zero every statically-dead gate's table
    dead_idx = jnp.asarray(np.flatnonzero(~live))
    t_pruned = t1
    for j in range(4):
        t_pruned = t_pruned.at[dead_idx * 4 + j].set(0)
    fwd = jax.vmap(net._forward, in_axes=(0, 0, 0, None))
    logits = fwd(net.codes(t1[None]), a1[None], b1[None], Xte_p)[0]
    logits_p = fwd(net.codes(t_pruned[None]), a1[None], b1[None], Xte_p)[0]
    pred, pred_p = logits.argmax(0), logits_p.argmax(0)
    acc = float((pred == yte).mean())
    acc_p = float((pred_p == yte).mean())
    identical = bool((pred == pred_p).all())
    logits_same = bool((logits == logits_p).all())

    full = net.model_memory_bytes()
    pruned = pruned_bytes(net, n_live)
    print(
        f"test acc: original {acc:.4f}  pruned {acc_p:.4f}  predictions identical: {identical}"
        f"  (integer logits identical: {logits_same})"
    )
    print(f"artifact: {full} B -> {pruned} B  ({full / pruned:.2f}x smaller)")
    metrics = {
        "test_acc": round(acc, 4),
        "n_gates": net.n_gates,
        "n_live": n_live,
        "model_memory_bytes_full": full,
        "model_memory_bytes_pruned": pruned,
        "predictions_identical": identical,
        "logits_identical": logits_same,
    }
    print("METRICS " + json.dumps(metrics))


if __name__ == "__main__":
    main(tyro.cli(PruneConfig))
