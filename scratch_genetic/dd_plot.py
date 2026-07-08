"""Double-descent plot for the NAND genetic net: LOSS (y) vs NUMBER OF NAND GATES (x, log scale).

Sweep the width from a handful of gates up to hundreds of millions (dd_sweep.sh / dd_sweep.sbatch),
train each width for a fixed wall-clock budget, then plot the FINAL train / val / test loss (loss =
1 - accuracy, in %) of each width against its gate count. If a double descent exists we should see
the test-loss curve dip, bump up around the interpolation region (#gates ~ #train samples), then
descend again as the net grows past it.

Reads every scratch_genetic/runs/dd_g*.jsonl (one per width; last line = final metrics).

    python scratch_genetic/dd_plot.py                       # -> scratch_genetic/runs/dd.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

N_TRAIN = 44992  # whole-word train set (interpolation threshold reference)


def best_record(path: Path) -> dict | None:
    """One row per width: gate count + the BEST (min) loss reached for each split over the run.
    Best-achieved loss (optimal early stopping) is used instead of the final eval because the GA
    optimizes train-batch margin, so test accuracy drifts after peaking and the last eval understates
    the model. min-loss across the run is the fair, low-noise capacity-vs-loss value."""
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not recs or recs[-1]["gen"] < 50:            # skip untrained runs (OOM/crash left a gen-0 row)
        return None
    out = {"gates": recs[-1]["gates"], "gen": recs[-1]["gen"]}
    for k in ("train_loss", "val_loss", "test_loss"):
        vals = [r[k] for r in recs if k in r]
        if vals:
            out[k] = min(vals)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, default=Path("scratch_genetic/runs"))
    p.add_argument("--glob", default="dd_g*.jsonl")
    p.add_argument("--out", type=Path, default=Path("scratch_genetic/runs/dd.png"))
    args = p.parse_args()

    rows = []
    for f in sorted(args.runs.glob(args.glob)):
        r = best_record(f)
        if r and "gates" in r:
            rows.append(r)
    rows.sort(key=lambda r: r["gates"])
    if not rows:
        raise SystemExit(f"no records in {args.runs}/{args.glob}")

    gates = [r["gates"] for r in rows]
    print(f"{len(rows)} widths, gates {gates[0]:,} .. {gates[-1]:,}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for key, color, label in [("train_loss", "#2c7fb8", "train loss"),
                              ("val_loss", "#d95f0e", "val loss"),
                              ("test_loss", "#31a354", "test loss")]:
        ys = [r.get(key) for r in rows]
        xs = [g for g, y in zip(gates, ys) if y is not None]
        ys = [y for y in ys if y is not None]
        ax.plot(xs, ys, "o-", color=color, label=label, lw=1.8, ms=5)

    ax.axvline(N_TRAIN, ls="--", color="gray", lw=1,
               label=f"#train ≈ {N_TRAIN:,} (interpolation ref)")
    ax.set_xscale("log")
    ax.set_xlabel("number of NAND gates  (depth × width, log scale)")
    ax.set_ylabel("loss = 1 − accuracy  (%)")
    ax.set_title("NAND genetic net: loss vs model size (double-descent sweep)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
