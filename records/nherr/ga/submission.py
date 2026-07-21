"""Our GA-trained LUT net, measured on the silicon axis (mnistbench: MNIST accuracy vs sky130 GE).

Our own size axis is `model_memory_bytes` -- the bits a deployment carries. That is the right axis
for lut-golf, but it prices nothing about the *circuit*: a thermometer comparator and a popcount
adder are free in bytes and real in silicon. mnistbench charges for both, so this record puts the GA
on an axis other people's methods already occupy.

Deliberate choices:

  * Their data, not ours. `mnistbench.data` holds the fixed 54k/6k/10k split and computes the test
    number itself by simulating the netlist. Reusing our loader would produce a number that only
    compares to itself.
  * Their thresholds. `even_thresholds` lands on 2^k-1 boundaries, so `pix > 127` is bit 7 of the
    byte -- a WIRE, zero gates. Our own default (32, 64, 96, ...) is one grey level off that and
    makes every threshold a real comparator, across 784 pixels, for no accuracy.
  * Their emitter. `hw.emit_lutnet` wants (idx_a, idx_b, tt) per layer, which is exactly what
    `Net.sources()` and `Net.codes()` already produce -- see emit_verilog() for the one real
    mismatch (signal numbering).

The circuit is the GA's own bit-exact forward pass, so predict() and the Verilog are the same
boolean function by construction; the harness rejects the point if they ever disagree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from mnistbench.data import N_CLASSES, Mnist
from mnistbench.hw import emit_lutnet, even_thresholds
from mnistbench.spec import Submission

# the GA sits next to this file; it is not a package, so load it by path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax.numpy as jnp  # noqa: E402
from ga_bits_wiring_mnist import Net, pack  # noqa: E402
from gate_importance_mnist import Config, train  # noqa: E402

TITLE = "genetic (learned truth tables + codebook wiring, pop 512 + crossover)"

# `m` is the configuration behind our 91.7% result; the others sweep it for a curve. Every point
# uses the same budget-matched GA (pop 512, gate-wise crossover, margin + 100*accuracy fitness).
POINTS = [
    {"name": "xs", "bits": 1, "widths": (512, 256, 160), "gens": 20000},
    {"name": "s", "bits": 3, "widths": (1536, 512, 320), "gens": 20000},
    {"name": "m", "bits": 7, "widths": (3072, 1024, 500), "gens": 20000},
    {"name": "l", "bits": 7, "widths": (6144, 2048, 1000), "gens": 20000},
]


class GaLutNet(Submission):
    def __init__(self, bits: int, widths: tuple[int, ...], gens: int, pop: int = 512) -> None:
        if widths[-1] % N_CLASSES:
            raise ValueError(f"readout width {widths[-1]} must be divisible by {N_CLASSES}")
        self.thresholds = even_thresholds(bits)
        self.cfg = Config(widths=list(widths), gens=gens, pop=pop, thresholds=self.thresholds)
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
        Xtr = self._encode(data.train_x)
        ytr = jnp.asarray(data.train_y, jnp.int32)
        self.net = Net(Xtr.shape[1], self.cfg.widths, N_CLASSES, codebook=self.cfg.wire_codebook)
        print(
            f"GA: {Xtr.shape[1]} input bits -> {list(self.cfg.widths)} "
            f"({self.net.n_gates} gates), pop {self.cfg.pop}, {self.cfg.gens} gens",
            flush=True,
        )
        tab, wa, wb = train(self.net, Xtr, ytr, self.cfg)  # sorted best-first
        self.tab, self.wa, self.wb = tab[0], wa[0], wb[0]

    # ---- the model, exactly as the circuit computes it -----------------------------------
    def _votes(self, pix: np.ndarray) -> np.ndarray:
        """(N, 784) uint8 -> (N, 10) integer class popcounts, from the GA's own bit-exact forward."""
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
        v = np.concatenate(out, 0)
        return v[:n]

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
    return GaLutNet(**point)
