"""DID -- discrete influence descent -- on the silicon axis.

One network, no population. Each sweep draws a fresh batch, linearises the loss into a signed
per-gate sensitivity, and turns that into a *ranked pool of proposals*: single truth-table rows,
parent-child pairs, and codebook rewires (a new source for one port, or both, each scored jointly
with the best-response table for the gate). The top of that one global ranking is then tried one at
a time by an exact forward pass, and a proposal is accepted only if the measured loss drops. Every
trial is charged against the same evaluation budget the GA gets, so the two searches are
comparable, not just similar.

The record this ports from measured DID on a *bytes* axis -- the bits a deployment carries. That
axis prices the genome and nothing else: the thermometer comparators and the popcount adder are
free in bytes and real in silicon. This puts the same search on the axis the rest of the board
occupies, where the encoder and the readout are charged too.

Two deliberate choices, both shared with `nherr/ga` so the comparison stays honest:

  * Their data. `mnistbench.data` holds the fixed 54k/6k/10k split and simulates the netlist for
    the test number. Nothing here ever touches `data.test_*`: the Runner's "test" slot -- the one
    it uses to keep the best genome seen -- is fed the *validation* set.
  * `even_thresholds`. Those land on 2^k-1 boundaries, so `pix > 127` is bit 7 of the byte, a
    wire. The upstream default (32, 64, 96, ...) is one grey level off that and would make all
    seven thresholds real comparators across 784 pixels, for no accuracy.

The circuit is the search's own bit-exact forward pass, so predict() and the Verilog are the same
boolean function by construction; the harness rejects the point if they ever disagree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from mnistbench.data import N_CLASSES, Mnist
from mnistbench.hw import emit_lutnet, even_thresholds
from mnistbench.spec import Submission

# the search sits next to this file; it is not a package, so load it by path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax.numpy as jnp  # noqa: E402
from evo_algos_mnist import Config, Runner  # noqa: E402
from ga_bits_wiring_mnist import Net, pack  # noqa: E402

TITLE = "did (discrete influence descent, codebook rewiring, no population)"

# `s` is the shape behind the headline number upstream (the 5,745 B class); the rest are the
# uniform shapes that record swept. Every point runs the same converged DID configuration --
# rewire + joint + dedup -- and differs only in the net it searches over.
POINTS = [
    {"name": "xs", "bits": 7, "widths": (620, 620, 600), "gens": 100000},
    {"name": "s", "bits": 7, "widths": (3072, 1024, 500), "gens": 100000},
    {"name": "m", "bits": 7, "widths": (2460, 2450, 2450), "gens": 100000},
    {"name": "l", "bits": 7, "widths": (4400, 4400, 4400), "gens": 100000},
    {"name": "xl", "bits": 7, "widths": (8800, 8800, 8800), "gens": 100000},
]


class DidLutNet(Submission):
    def __init__(self, bits: int, widths: tuple[int, ...], gens: int, pop: int = 512) -> None:
        if widths[-1] % N_CLASSES:
            raise ValueError(f"readout width {widths[-1]} must be divisible by {N_CLASSES}")
        self.thresholds = even_thresholds(bits)
        self.cfg = Config(
            algo="did",
            widths=list(widths),
            thresholds=list(self.thresholds),
            gens=gens,
            pop=pop,
            did_rewire=True,  # codebook topology moves in the same ranked pool as table rows
            did_joint=True,  # score both ports moving together, not only one at a time
            did_dedup=True,  # one trial per written gate; the rest would run on a stale genome
        )
        self.net: Net | None = None
        self.tab = self.wa = self.wb = None

    # ---- encoding: must match hw.emit_thermometer bit-for-bit ---------------------------
    def _encode(self, pix: np.ndarray) -> jnp.ndarray:
        """(N, 784) uint8 -> (N, 784*k) 0/1, pixel-major: bit p*k + j is pix[p] > thresholds[j].

        Same layout as the emitted encoder, so a gate's source index means the same thing in the
        python model and in the Verilog.
        """
        t = jnp.asarray(self.thresholds, jnp.int32)
        x = jnp.asarray(pix, jnp.int32).reshape(pix.shape[0], -1, 1)
        return (x > t).astype(jnp.uint8).reshape(pix.shape[0], -1)

    def train(self, data: Mnist, *, device: str = "cpu", seed: int = 0) -> None:
        self.cfg.seed = seed
        Xtr, ytr = self._encode(data.train_x), jnp.asarray(data.train_y, jnp.int32)
        Xva, yva = self._encode(data.val_x), jnp.asarray(data.val_y, jnp.int32)
        self.net = Net(Xtr.shape[1], self.cfg.widths, N_CLASSES, codebook=self.cfg.wire_codebook)
        print(
            f"DID: {Xtr.shape[1]} input bits -> {list(self.cfg.widths)} "
            f"({self.net.n_gates} gates), {self.cfg.gens} sweeps x {self.cfg.pop} evals",
            flush=True,
        )
        # the Runner's held-out slot picks which genome to keep -- give it validation, never test
        runner = Runner(self.net, Xtr, ytr, Xva, yva, self.cfg)
        runner.run()
        genome = runner.best[0] if runner.best[0] is not None else runner.cur_best()
        self.tab, self.wa, self.wb = genome

    # ---- the model, exactly as the circuit computes it -----------------------------------
    def _votes(self, pix: np.ndarray) -> np.ndarray:
        """(N, 784) uint8 -> (N, 10) integer class popcounts, from the search's own forward."""
        enc = self._encode(pix)
        n = enc.shape[0]
        padded = (-n) % 8  # the packed forward works on 8 samples per byte
        if padded:
            enc = jnp.concatenate([enc, jnp.zeros((padded, enc.shape[1]), enc.dtype)], 0)
        out = []
        for i in range(0, enc.shape[0], 8192):
            chunk = enc[i : i + 8192]
            logits = self.net._forward(
                self.net.codes(self.tab[None])[0], self.wa, self.wb, pack(chunk)
            )
            out.append(np.asarray(logits).T)  # (chunk, 10)
        return np.concatenate(out, 0)[:n]

    def predict(self, pix: np.ndarray) -> np.ndarray:
        # ties -> lowest class, matching the emitted argmax and numpy's argmax
        return self._votes(pix).argmax(1)

    def scores(self, pix: np.ndarray) -> np.ndarray:
        """Per-class firing fraction in [0, 1]: the popcounts over the gates per group."""
        return self._votes(pix) / (self.net.widths[-1] // N_CLASSES)

    def emit_verilog(self) -> str:
        """Our genome -> their emitter.

        The one real mismatch: their signal ids are GLOBAL (the encoder owns 0..n_in-1, then every
        gate in order), while our wires are per-layer indices into the previous layer's outputs. So
        each layer's sources get that layer's base offset added. Truth tables need no translation --
        both sides encode bit (2a+b) = f(a, b).
        """
        net = self.net
        sa, sb = (np.asarray(s) for s in net.sources(self.wa, self.wb))
        code = np.asarray(net.codes(self.tab[None])[0])

        layers = []
        for k in range(len(net.widths)):
            lo, hi = net.offs[k], net.offs[k + 1]
            base = 0 if k == 0 else net.n_in + net.offs[k - 1]  # layer 0 reads the encoder itself
            layers.append((base + sa[lo:hi], base + sb[lo:hi], code[lo:hi]))
        return emit_lutnet(self.thresholds, layers, n_classes=N_CLASSES)


def build(**point) -> Submission:
    return DidLutNet(**point)
