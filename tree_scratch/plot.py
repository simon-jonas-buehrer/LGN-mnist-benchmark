"""Plot train + val accuracy curves for the two tree methods (boost vs conv_boost).

Reads the incrementally-written .jsonl logs (one record per boosting round, with train/val/test
accuracy and a wall-clock 'min' field) and draws train (dashed) + val (solid) vs boosting round
for each method on shared axes. Test accuracy is annotated at the final point.

    python tree_scratch/plot.py                       # defaults: dense=b_dense, conv=c_conv
    python tree_scratch/plot.py --runs dense=tree_scratch/runs/b_dense conv=tree_scratch/runs/c_conv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]


def load(prefix: Path) -> list[dict]:
    f = prefix.with_suffix(".jsonl")
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+",
                   default=["dense=tree_scratch/runs/b_dense",
                            "dense+aug=tree_scratch/runs/dense_aug",
                            "conv=tree_scratch/runs/c_conv",
                            "conv+aug=tree_scratch/runs/push_conv"],
                   help="label=prefix pairs (prefix without extension)")
    p.add_argument("--out", type=Path, default=Path("tree_scratch/runs/curves.png"))
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, spec in enumerate(args.runs):
        label, prefix = spec.split("=", 1)
        recs = load(Path(prefix))
        if not recs:
            print(f"[skip] no data for {label} ({prefix}.jsonl)")
            continue
        x = [r["tree"] for r in recs]
        color = PALETTE[i % len(PALETTE)]
        ax.plot(x, [r["train"] for r in recs], "--", color=color, alpha=0.6,
                label=f"{label} train")
        ax.plot(x, [r["val"] for r in recs], "-", color=color, linewidth=2,
                label=f"{label} val")
        last = recs[-1]
        note = f"val {last['val']:.1f}" + (f" / test {last['test']:.1f}" if "test" in last else "")
        ax.annotate(note, (x[-1], last["val"]), fontsize=9, color=color,
                    xytext=(5, 0), textcoords="offset points", va="center")
        print(f"{label}: {len(recs)} rounds, final train {last['train']:.2f} "
              f"val {last['val']:.2f}" + (f" test {last['test']:.2f}" if "test" in last else ""))

    ax.set_xlabel("boosting round (tree index)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Tree methods on CIFAR-10: augmentation × conv-locality levers "
                 "(train dashed, val solid)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
