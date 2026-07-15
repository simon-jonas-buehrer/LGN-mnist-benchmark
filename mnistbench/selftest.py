"""End-to-end self-test of the harness on a tiny random (untrained) net.

    python -m mnistbench.selftest

It checks the property the benchmark rests on: the Verilog we emit, the netlist yosys and ABC
give back, and a plain python forward pass are the same boolean function. If this passes, a gap
between two records is a gap between two optimizers, not between two emission bugs.

The accuracy here is irrelevant (a random net scores ~10%). What is checked is bit-exact
agreement on every image, and that a real area number comes out.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from . import hw, netlist, synth
from .data import N_CLASSES, load, to_bits

N_SAMPLES = 256


def reference(enc: np.ndarray, layers, n_classes: int = N_CLASSES) -> np.ndarray:
    """Evaluate a fan-in-2 LUT net in python, wired exactly as hw.emit_lutnet() wires it."""
    sig = enc.astype(np.uint8)
    for idx_a, idx_b, tt in layers:
        sel = 2 * sig[:, idx_a] + sig[:, idx_b]  # index into the truth table
        out = (tt >> sel) & 1  # bit (2a+b) of tt
        sig = np.concatenate([sig, out.astype(np.uint8)], axis=1)
    last = sig[:, -len(layers[-1][0]):]
    counts = last.reshape(len(sig), n_classes, -1).sum(-1)
    return counts.argmax(1)  # ties -> lowest class, same as the argmax we emit


def main() -> None:
    rng = np.random.default_rng(0)
    thresholds = [63, 127, 191]
    n_in = 784 * len(thresholds)

    widths, layers, off = [64, 40], [], n_in  # the last width must be divisible by 10
    for w in widths:
        layers.append((rng.integers(off, size=w), rng.integers(off, size=w),
                       rng.integers(16, size=w)))
        off += w

    sv_text = hw.emit_lutnet(thresholds, layers)
    print(f"emitted {len(sv_text.splitlines())} lines of verilog")

    pix = load().test_x[:N_SAMPLES]
    enc = (pix[:, :, None] > np.array(thresholds)).reshape(N_SAMPLES, n_in)  # pixel-major
    py = reference(enc, layers)

    with tempfile.TemporaryDirectory() as td:
        sv = Path(td) / "top.sv"
        sv.write_text(sv_text)
        area = synth.synth_area(sv)
        print(f"area : {area.ge:,.1f} GE  ({area.area_um2:,.0f} um^2, {area.cells:,} cells)")
        nand = synth.synth_nand(sv)
        net = netlist.from_json(nand.netlist)
        print(f"nand : {nand.gates:,} gates ({nand.nand:,} NAND + {nand.inv:,} INV), "
              f"depth {net.depth}")
        sim = netlist.run(net, to_bits(pix))

    if not (sim == py).all():
        bad = np.flatnonzero(sim != py)[:8]
        raise SystemExit(f"FAIL: disagree at {bad.tolist()}: circuit {sim[bad].tolist()} "
                         f"vs python {py[bad].tolist()}")
    if area.ge <= 0:
        raise SystemExit("FAIL: no area reported")
    print(f"OK -- emit / synthesize / simulate agree bit for bit on {N_SAMPLES} images")


if __name__ == "__main__":
    main()
