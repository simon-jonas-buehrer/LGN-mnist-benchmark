"""Plot tree DOUBLE DESCENT from dd.py output: misclassification error vs #parameters.

Reads dd.jsonl (records tagged regime=classical|modern, with params=#leaves in the ensemble and
train/val/test accuracy) and plots ERROR = 100 - accuracy vs params on a log-x axis. The classical
regime (one growing tree) and the modern regime (growing bagged interpolating forest) share the
x-axis, so the full double descent is visible: error falls, PEAKS at the interpolation threshold
(single tree just fits train), then falls AGAIN as bagged trees average out the variance.

    python tree_scratch/dd_plot.py                       # default: tree_scratch/runs/dd.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=Path("tree_scratch/runs/dd"),
                   help="prefix of the dd.py run (reads <prefix>.jsonl)")
    p.add_argument("--out", type=Path, default=Path("tree_scratch/runs/double_descent.png"))
    p.add_argument("--metric", choices=["val", "test"], default="val",
                   help="which error curve to headline (train always shown)")
    args = p.parse_args()

    recs = [json.loads(x) for x in args.run.with_suffix(".jsonl").read_text().splitlines()
            if x.strip()]
    if not recs:
        print("no records")
        return
    cls = [r for r in recs if r["regime"] == "classical"]
    mod = [r for r in recs if r["regime"] == "modern"]

    def err(rows, key):
        return [r["params"] for r in rows], [100 - r[key] for r in rows]

    fig, ax = plt.subplots(figsize=(9.5, 6))
    # train error (both regimes) — light, to show the interpolation threshold (train err -> 0)
    for rows in (cls, mod):
        if rows:
            x, y = err(rows, "train")
            ax.plot(x, y, "--", color="#bbbbbb", linewidth=1.2, zorder=1)
    ax.plot([], [], "--", color="#bbbbbb", label="train error")

    if cls:
        x, y = err(cls, args.metric)
        ax.plot(x, y, "o-", color="#1f77b4", linewidth=2, markersize=5,
                label=f"{args.metric} error — classical (1 tree, grow leaves)")
    if mod:
        x, y = err(mod, args.metric)
        ax.plot(x, y, "o-", color="#d62728", linewidth=2, markersize=4,
                label=f"{args.metric} error — modern (bagged interpolating forest)")

    # interpolation threshold = first classical point with train error ~ 0
    interp = next((r["params"] for r in cls if r["train"] >= 99.5), None)
    if interp:
        ax.axvline(interp, color="#2ca02c", alpha=0.4, linestyle=":")
        ax.annotate("interpolation\nthreshold\n(train err ~0)", (interp, ax.get_ylim()[1]),
                    fontsize=9, color="#2ca02c", xytext=(6, -6), textcoords="offset points",
                    va="top")

    allrows = cls + mod
    for tag, rows in (("classical", cls), ("modern", mod)):
        if rows:
            b = min(rows, key=lambda r: 100 - r[args.metric])
            print(f"{tag:9s} best {args.metric} error {100 - b[args.metric]:.2f} "
                  f"({args.metric} acc {b[args.metric]:.2f}) @ params {b['params']}")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("model size:  total #leaves in the ensemble  (~ #parameters)")
    ax.set_ylabel("error = 100 - accuracy  (%)")
    ax.set_title("Tree double descent on CIFAR-10 (single-tree capacity  →  bagged forest size)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower center", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
