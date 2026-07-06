"""Plot the optimizer-benchmark learning curves from scratch/runs/*.jsonl.

Runs are named <method>_conn<0|1>_s<seed>.jsonl; the 3 seeds of each of the 8
configurations are aggregated into one curve (mean, with a min-max band). Panels:
rows = train / val, columns = loss / accuracy / perplexity. Two figures, identical
layout, different x-axis:

    curves.png        x = samples seen        (optimizer sample efficiency)
    curves_time.png   x = wall-clock hours    (what a GPU-day actually buys)

Color = method, solid = learnable connections, dashed = frozen monarch wiring.

    .venv/bin/python scratch/plot.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = Path(__file__).parent / "runs"
COLORS = {"bp": "tab:blue", "cd": "tab:orange", "rs": "tab:green", "mab": "tab:red"}
METRICS = [("loss", "cross-entropy loss"), ("acc", "accuracy (%)"), ("ppl", "perplexity")]


def draw(groups: dict, xval, xlabel: str, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for cfg, seeds in sorted(groups.items()):
        color = COLORS.get(cfg.split("_")[0], "gray")
        ls = "--" if "conn0" in cfg else "-"
        # common x grid: interpolate every seed onto it, then mean / min / max
        lo = max(min(xval(r) for r in s) for s in seeds)
        hi = min(max(xval(r) for r in s) for s in seeds)
        if hi <= lo:
            continue
        grid = np.geomspace(max(lo, 1e-3), hi, 80)
        for i, split in enumerate(("train", "val")):
            for j, (key, _) in enumerate(METRICS):
                ys = np.stack([np.interp(grid, [xval(r) for r in s],
                                         [r[split][key] for r in s]) for s in seeds])
                ax = axes[i][j]
                ax.plot(grid, ys.mean(0), color=color, ls=ls, lw=1.6,
                        label=f"{cfg} ({len(seeds)}s)" if (i, j) == (0, 0) else None)
                if len(seeds) > 1:
                    ax.fill_between(grid, ys.min(0), ys.max(0), color=color, alpha=0.15)
    for i, split in enumerate(("train", "val")):
        for j, (key, title) in enumerate(METRICS):
            ax = axes[i][j]
            ax.set_title(f"{split} {title}")
            ax.set_xscale("log")
            ax.grid(alpha=0.3)
            if key == "ppl":
                ax.set_yscale("log")
            if i == 1:
                ax.set_xlabel(xlabel)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Optimizers on a fixed monarch-wired LUT net (depth 8 x 64K, fan-in 4), "
                 "CIFAR-10, mean over seeds (band = min-max)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    groups: dict[str, list[list[dict]]] = defaultdict(list)  # config -> per-seed curves
    for f in sorted(RUNS.glob("*.jsonl")):
        rows = [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]
        rows = [r for r in rows if r["samples"] > 0]
        if rows:
            cfg = f.stem.rsplit("_s", 1)[0] if "_s" in f.stem else f.stem
            groups[cfg].append(rows)
    if not groups:
        sys.exit(f"no .jsonl runs in {RUNS}")
    draw(groups, lambda r: r["samples"], "samples seen", RUNS / "curves.png")
    draw(groups, lambda r: r["min"] / 60.0, "wall-clock hours (1 GPU)",
         RUNS / "curves_time.png")


if __name__ == "__main__":
    main()
