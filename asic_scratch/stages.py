"""Measure the three-stage shrinkage on one conv kernel.

    stage 0  generic     weights as runtime inputs   -> real int8 multipliers
    stage 1  folded      weights baked in as consts  -> multipliers become a shift-add tree
    stage 2  folded+opt  ABC minimization on top     -> rewriting/refactoring/resub

Usage:  python stages.py --taps 27            # RepVGG stage-0 kernel (3x3x3)
        python stages.py --taps 27 --sparsity 0.5
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path

from emit_sv import Kernel, emit_generic_kernel, emit_kernel
from synth import BASELINE, OPTIMIZED, synth


def measure(taps: int, sparsity: float = 0.0, seed: int = 0, skip_generic: bool = False) -> dict:
    rng = random.Random(seed)
    w = [0 if rng.random() < sparsity else rng.randint(-127, 127) for _ in range(taps)]
    k = Kernel(weights=w, bias=rng.randint(-1000, 1000), mult=rng.randint(1 << 28, 1 << 30), shift=30)

    out: dict[str, object] = {"taps": taps, "sparsity": sparsity, "nonzero": sum(x != 0 for x in w)}
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)

        if not skip_generic:
            g = d / "generic.sv"
            g.write_text(emit_generic_kernel(taps))
            out["generic"] = synth(g, "kernel_generic", BASELINE).gates

        f = d / "folded.sv"
        f.write_text(emit_kernel(k))
        out["folded"] = synth(f, "kernel", BASELINE).gates
        out["folded_opt"] = synth(f, "kernel", OPTIMIZED).gates

    if not skip_generic:
        out["fold_ratio"] = round(out["generic"] / out["folded"], 2)
    out["opt_ratio"] = round(out["folded"] / out["folded_opt"], 2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--taps", type=int, default=27)
    ap.add_argument("--sparsity", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-generic", action="store_true")
    a = ap.parse_args()
    print(json.dumps(measure(a.taps, a.sparsity, a.seed, a.skip_generic), indent=2))
