"""The harness: run a record's points, measure each one, write results.json.

Per point:

    build -> train (timed) -> emit_verilog
          -> yosys/ABC -> sky130 cells   -> area / area(NAND2) = GE          [x-axis]
          -> yosys/ABC -> NAND netlist   -> simulate on 10k test images      [y-axis]
          -> cross-check the netlist against the submission's own predict()

The cross-check is a hard failure, not a warning: if a submission's model and its circuit
disagree, one of the two numbers on the leaderboard is a lie and we do not know which.
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path
from types import ModuleType

import numpy as np

from . import netlist, synth
from .data import Mnist, to_bits

CHECK_SAMPLES = 512


def _cross_entropy(logits: np.ndarray, y: np.ndarray) -> float:
    """Mean softmax cross-entropy of (N, 10) logits against labels y."""
    z = logits - logits.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return float(-logp[np.arange(len(z)), y].mean())


def _fit_temperature(scores: np.ndarray, y: np.ndarray) -> float:
    """Scalar T > 0 minimising CE(scores / T, y). Coarse-to-fine over log T; CE is unimodal in T,
    and any T > 0 preserves the argmax, so calibration never changes the predicted class."""
    lo, hi = 1e-3, 1e2
    for _ in range(6):  # ~40x zoom per pass, converges to a tight bracket
        grid = np.geomspace(lo, hi, 40)
        ces = [_cross_entropy(scores / t, y) for t in grid]
        j = int(np.argmin(ces))
        lo, hi = grid[max(0, j - 1)], grid[min(len(grid) - 1, j + 1)]
    return float(np.sqrt(lo * hi))


def load_record(path: Path) -> ModuleType:
    sub = path / "submission.py"
    if not sub.exists():
        raise SystemExit(f"{sub} not found -- a record is a directory containing submission.py")
    spec = importlib.util.spec_from_file_location(f"record_{path.name}", sub)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for attr in ("POINTS", "build"):
        if not hasattr(mod, attr):
            raise SystemExit(f"{sub} must define {attr} (see mnistbench/spec.py)")
    return mod


def measure(sv: Path, data: Mnist) -> tuple[dict, netlist.NandNet]:
    """Both axes, from the Verilog alone: sky130 area (x) and netlist accuracy (y).

    Returns the metrics (JSON-serializable) and the simulated netlist, which the caller uses to
    cross-check the submission's own predict().
    """
    t0 = time.time()
    area = synth.synth_area(sv)
    print(f"[area ] {area.ge:,.0f} GE  ({area.area_um2:,.0f} um^2, {area.cells:,} cells, "
          f"{time.time() - t0:.0f}s)", flush=True)

    t0 = time.time()
    nand = synth.synth_nand(sv)
    net = netlist.from_json(nand.netlist)
    print(f"[nand ] {nand.gates:,} gates ({nand.nand:,} NAND + {nand.inv:,} INV), "
          f"depth {net.depth}, {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    pred = netlist.run(net, to_bits(data.test_x))
    test_acc = float((pred == data.test_y).mean()) * 100
    print(f"[sim  ] test acc {test_acc:.2f}%  (from the netlist, {time.time() - t0:.0f}s)",
          flush=True)

    return {"ge": round(area.ge, 1), "area_um2": round(area.area_um2, 1), "cells": area.cells,
            "nand": nand.nand, "inv": nand.inv, "depth": net.depth,
            "test_acc": round(test_acc, 2)}, net


def rescore_record(record: Path, data: Mnist) -> None:
    """Re-measure stored .sv artifacts without retraining (both axes come from the Verilog)."""
    path = record / "results.json"
    results = json.loads(path.read_text())
    for p in results["points"]:
        sv = record / "artifacts" / f"{p['name']}.sv"
        if not sv.exists():
            print(f"=== {p['name']}: no artifact, skipping", flush=True)
            continue
        print(f"\n=== {p['name']} (rescore)", flush=True)
        m, _ = measure(sv, data)
        p.update(m)
        path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"[write] {path}", flush=True)


def run_point(mod: ModuleType, point: dict, data: Mnist, *, device: str, seed: int,
              artifacts: Path) -> dict:
    cfg = {k: v for k, v in point.items() if k != "name"}
    print(f"\n=== {point['name']}  {cfg}", flush=True)
    model = mod.build(**cfg)

    t0 = time.time()
    model.train(data, device=device, seed=seed)
    train_s = time.time() - t0
    print(f"[train] {train_s:.0f}s", flush=True)

    artifacts.mkdir(parents=True, exist_ok=True)
    sv = artifacts / f"{point['name']}.sv"
    sv.write_text(model.emit_verilog())
    print(f"[emit ] {sv.name}, {sv.stat().st_size / 1e6:.1f} MB", flush=True)

    m, net = measure(sv, data)

    # the circuit and the submission's own model must be the same boolean function
    hw_check = netlist.run(net, to_bits(data.test_x[:CHECK_SAMPLES]))
    py_check = np.asarray(model.predict(data.test_x[:CHECK_SAMPLES]))
    if not (hw_check == py_check).all():
        bad = np.flatnonzero(hw_check != py_check)
        raise SystemExit(
            f"REJECTED: circuit and model disagree on {len(bad)}/{CHECK_SAMPLES} test images "
            f"(e.g. {bad[:5].tolist()}: circuit says {hw_check[bad[:5]].tolist()}, model says "
            f"{py_check[bad[:5]].tolist()}).\n"
            "emit_verilog() must be the exact function predict() computes."
        )

    val_acc = float((np.asarray(model.predict(data.val_x)) == data.val_y).mean()) * 100
    out = {**point, **m, "val_acc": round(val_acc, 2), "train_s": round(train_s),
           "device": device, "seed": seed}

    # cross-entropy over the readout's per-class firing fractions (see Submission.scores). Faithful
    # to the circuit: the fractions are the popcount groups the netlist computes, and their argmax
    # is the class the netlist emits -- the equivalence check above already pins that down.
    sc_va = model.scores(data.val_x)
    sc_te = model.scores(data.test_x)
    if sc_te is not None:
        # The raw fractions live in a narrow band (a NAND net's gates mostly fire), so softmax over
        # them is near-uniform and CE would read ~ln(10) no matter the accuracy. So temperature-
        # scale: fit ONE scalar T on val to minimise CE, then report test CE at that T. T > 0 can't
        # move an argmax, so the calibrated class is still exactly the circuit's -- this is the
        # circuit's own votes, reported at their best-calibrated confidence, not an invented signal.
        t = _fit_temperature(np.asarray(sc_va, float), data.val_y)
        out["test_ce"] = round(_cross_entropy(np.asarray(sc_te, float) / t, data.test_y), 4)
        out["ce_temp"] = round(float(t), 4)
    return out


def merge_record(record: Path) -> None:
    """Assemble results.json from the per-point files. Safe to run any time; needed after a
    parallel run, where each point was measured by its own process."""
    mod = load_record(record)
    done = {}
    for f in (record / "artifacts").glob("*.point.json"):
        p = json.loads(f.read_text())
        done[p["name"]] = p
    if not done:
        raise SystemExit(f"no measured points under {record}/artifacts")
    results = {
        "record": f"{record.parent.name}/{record.name}",
        "title": getattr(mod, "TITLE", record.name),
        "points": [done[p["name"]] for p in mod.POINTS if p["name"] in done],
    }
    path = record / "results.json"
    tmp = path.with_suffix(f".json.{os.getpid()}")  # atomic: a concurrent reader never sees a
    tmp.write_text(json.dumps(results, indent=2) + "\n")  # half-written file
    tmp.replace(path)
    print(f"[merge] {path}: {len(results['points'])} points "
          f"({', '.join(p['name'] for p in results['points'])})", flush=True)


def run_record(record: Path, data: Mnist, *, device: str, seed: int, only: list[str] | None,
               force: bool) -> None:
    mod = load_record(record)
    artifacts = record / "artifacts"

    for point in mod.POINTS:
        if only and point["name"] not in only:
            continue
        # one file per point: a point is written only by the process that measured it, so points
        # can be measured in parallel (one slurm job each) without racing on a shared results.json
        pf = artifacts / f"{point['name']}.point.json"
        if pf.exists() and not force:
            print(f"=== {point['name']}: already measured, skipping (--force to redo)", flush=True)
            continue
        res = run_point(mod, point, data, device=device, seed=seed, artifacts=artifacts)
        pf.write_text(json.dumps(res, indent=2) + "\n")
        print(f"[write] {pf}", flush=True)

    merge_record(record)
