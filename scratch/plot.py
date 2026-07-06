"""Plot the optimizer-benchmark learning curves from scratch/runs/*.jsonl.

Runs are named <method>_conn<0|1>_s<seed>.jsonl; the 3 seeds of each of the 8
configurations are aggregated into one curve (mean, with a min-max band). Panels:
rows = train / val, columns = loss / accuracy / perplexity; x-axis = samples seen (log).
Color = method, solid = learnable connections, dashed = frozen monarch wiring.

    .venv/bin/python scratch/plot.py                # -> scratch/runs/curves.png
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

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    metrics = [("loss", "cross-entropy loss"), ("acc", "accuracy (%)"),
               ("ppl", "perplexity")]
    for cfg, seeds in sorted(groups.items()):
        method = cfg.split("_")[0]
        color = COLORS.get(method, "gray")
        ls = "--" if "conn0" in cfg else "-"
        # common samples grid: interpolate every seed onto it, then mean / min / max
        lo = max(min(r["samples"] for r in s) for s in seeds)
        hi = min(max(r["samples"] for r in s) for s in seeds)
        grid = np.geomspace(max(lo, 1), max(hi, 2), 80)
        for i, split in enumerate(("train", "val")):
            for j, (key, _) in enumerate(metrics):
                ys = np.stack([np.interp(grid, [r["samples"] for r in s],
                                         [r[split][key] for r in s]) for s in seeds])
                ax = axes[i][j]
                ax.plot(grid, ys.mean(0), color=color, ls=ls, lw=1.6,
                        label=f"{cfg} ({len(seeds)}s)" if (i, j) == (0, 0) else None)
                if len(seeds) > 1:
                    ax.fill_between(grid, ys.min(0), ys.max(0), color=color, alpha=0.15)
    for i, split in enumerate(("train", "val")):
        for j, (key, title) in enumerate(metrics):
            ax = axes[i][j]
            ax.set_title(f"{split} {title}")
            ax.set_xscale("log")
            ax.grid(alpha=0.3)
            if key == "ppl":
                ax.set_yscale("log")
            if i == 1:
                ax.set_xlabel("samples seen")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Optimizers on a fixed monarch-wired LUT net (depth 8 x 64K, fan-in 4), "
                 "CIFAR-10, mean over seeds (band = min-max)")
    fig.tight_layout()
    out = RUNS / "curves.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
