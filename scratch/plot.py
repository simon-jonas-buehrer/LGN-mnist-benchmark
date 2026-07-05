"""Plot / animate a cd.py run from its per-round stats.

Reads <prefix>.jsonl (distributions, written by cd.py when --ckpt is set) and <prefix>.out
(curve rows, also covers rounds before stats logging existed). Writes:

    <prefix>_plots.png      static overview: curves, moves/round, and the three
                            distributions over time as heatmaps
    <prefix>_anim.gif       (--gif) animated bar charts of the distributions per round

    .venv/bin/python scratch/plot.py scratch/cd1280
    .venv/bin/python scratch/plot.py scratch/cd1280 --gif
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(prefix: Path) -> list[dict]:
    recs: dict[int, dict] = {}
    out = prefix.with_suffix(".out")
    if out.exists():
        for line in out.read_text(errors="replace").splitlines():
            if re.match(r"\s*\d+ \|", line) and line.count("|") >= 9:
                f = [p.strip() for p in line.split("|")]
                s = 1 if line.count("|") >= 10 else 0                # newer rows have an rs col
                r = int(f[0])
                recs[r] = {"round": r, "ttbits": int(f[1]), "rewires": int(f[2]),
                           "shares": int(f[3]), "train": float(f[4 + s]),
                           "hinge": float(f[5 + s]),
                           "val": None if f[6 + s] == "nan" else float(f[6 + s]),
                           "test": None if f[7 + s] == "nan" else float(f[7 + s])}
    jl = prefix.with_suffix(".jsonl")
    if jl.exists():
        for line in jl.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                recs[d["round"]] = {**recs.get(d["round"], {}), **d}
    return [recs[k] for k in sorted(recs)]


def curve(recs, key):
    pts = [(r["round"], r[key]) for r in recs if r.get(key) is not None]
    return [p[0] for p in pts], [p[1] for p in pts]


def heat(ax, recs, key, title, ylabel, norm_rows=False):
    rs = [r for r in recs if r.get(key) is not None]
    if not rs:
        ax.set_title(f"{title} (no stats yet)")
        return
    m = np.array([r[key] for r in rs], dtype=float).T                # (bins, rounds)
    m /= np.maximum(m.sum(0, keepdims=True), 1)                      # fraction of gates
    x = [r["round"] for r in rs]
    im = ax.imshow(np.log10(m + 1e-7), aspect="auto", origin="lower", cmap="viridis",
                   extent=(x[0] - 0.5, x[-1] + 0.5, -0.5, m.shape[0] - 0.5))
    plt.colorbar(im, ax=ax, label="log10 fraction of gates")
    ax.set_title(title); ax.set_xlabel("round"); ax.set_ylabel(ylabel)


def churn(recs, key):
    """Per-round churn RATE = accepted moves of a lever / live gates (fraction of the network
    that lever rewrote that round)."""
    xs, ys = [], []
    for r in recs:
        g = r.get("gates")
        if r.get(key) is not None and g:
            xs.append(r["round"]); ys.append(r[key] / g)
    return xs, ys


def static_plots(recs, out_path: Path) -> None:
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold"})
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(15, 6))

    # (left) accuracy -- from round 0 (random baseline) so the climb from chance is visible
    for key, c in (("train", "tab:blue"), ("val", "tab:orange"), ("test", "tab:green")):
        x, y = curve(recs, key)
        a0.plot(x, y, label=key, color=c, lw=2.2, marker="o", ms=4)
        if y:
            a0.annotate(f"{y[-1]:.1f}%", (x[-1], y[-1]), color=c, fontsize=11, fontweight="bold",
                        xytext=(6, 0), textcoords="offset points", va="center")
    a0.axhline(10, color="gray", ls=":", lw=1, zorder=0)
    a0.text(0.02, 10.5, "chance (10%)", color="gray", fontsize=9, transform=a0.get_yaxis_transform())
    a0.set_title("accuracy: learning from random"); a0.set_xlabel("round (epoch)")
    a0.set_ylabel("accuracy  %"); a0.set_ylim(0, 100)
    a0.legend(loc="lower right"); a0.grid(alpha=0.3)

    # (right) learning activity -- churn rate per lever, from round 1 (round 0 has no moves)
    act = [r for r in recs if r["round"] >= 1]
    for key, c, lab in (("ttbits", "tab:blue", "truth-table churn"),
                        ("rewires", "tab:red", "connection churn"),
                        ("shares", "tab:purple", "sharing churn"),
                        ("clsmoves", "tab:green", "output-class churn"),
                        ("rebuilds", "tab:brown", "rebuild churn")):
        x, y = churn(act, key)
        if any(y):
            a1.plot(x, y, label=lab, color=c, lw=2, marker="o", ms=3)
    a1.set_yscale("log")
    a1.set_title("learning activity: accepted edits / gate per round")
    a1.set_xlabel("round (epoch)"); a1.set_ylabel("churn rate  (edits per gate)")
    a1.legend(loc="upper right", fontsize=10); a1.grid(alpha=0.3, which="both")

    last = recs[-1]
    fig.suptitle(f"backprop-free LUT-gate network  —  round {last['round']}: "
                 f"train {last.get('train')}%   val {last.get('val')}%",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


def animation(recs, out_path: Path) -> None:
    from matplotlib.animation import FuncAnimation, PillowWriter
    rs = [r for r in recs if r.get("ttbit_ones") is not None]
    if not rs:
        print("no distribution stats yet -- skipping gif")
        return
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    (a_bit, a_pop), (a_cp, a_acc) = ax

    def update(i):
        r = rs[i]
        for a in (a_bit, a_pop, a_cp, a_acc):
            a.clear()
        g = max(1, r.get("gates", 1))
        a_bit.bar(range(len(r["ttbit_ones"])), np.array(r["ttbit_ones"]) / g,
                  color="tab:blue")
        a_bit.set_ylim(0, 1); a_bit.axhline(0.5, color="gray", lw=0.5)
        a_bit.set_title("P(bit=1) per truth-table cell"); a_bit.set_xlabel("cell")

        a_pop.bar(range(len(r["ttpop_hist"])), np.array(r["ttpop_hist"]) / g,
                  color="tab:green")
        a_pop.set_title("bits set per gate"); a_pop.set_xlabel("# bits set")

        cp = np.array(r["copies_hist"], dtype=float)
        a_cp.bar(range(len(cp)), np.maximum(cp, 0.5), color="tab:purple", log=True)
        a_cp.set_title("copies per gate (log2 bins)"); a_cp.set_xlabel("log2 copies")
        a_cp.set_ylim(0.5, cp.max() * 2)

        j = [x for x in recs if x["round"] <= r["round"]]
        for key, c in (("train", "tab:blue"), ("val", "tab:orange"), ("test", "tab:green")):
            a_acc.plot(*curve(j, key), label=key, color=c)
        a_acc.set_xlim(recs[0]["round"], recs[-1]["round"]); a_acc.set_ylim(0, 100)
        a_acc.legend(loc="lower right"); a_acc.grid(alpha=0.3)
        a_acc.set_title(f"round {r['round']}   train={r.get('train')}  val={r.get('val')}")
        fig.suptitle("pure CD on a random LUT window: 3-lever training", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return []

    anim = FuncAnimation(fig, update, frames=len(rs), blit=False)
    anim.save(str(out_path), writer=PillowWriter(fps=4))
    plt.close(fig)
    print(f"wrote {out_path}  ({len(rs)} frames)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prefix", type=Path, help="run prefix, e.g. scratch/cd1280")
    p.add_argument("--gif", action="store_true")
    args = p.parse_args()
    recs = load(args.prefix)
    if not recs:
        raise SystemExit(f"no rounds found for {args.prefix}(.out/.jsonl)")
    static_plots(recs, args.prefix.with_name(args.prefix.name + "_plots.png"))
    if args.gif:
        animation(recs, args.prefix.with_name(args.prefix.name + "_anim.gif"))


if __name__ == "__main__":
    main()
