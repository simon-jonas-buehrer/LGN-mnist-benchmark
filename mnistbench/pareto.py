"""Collect every record's results.json into the plots and the leaderboard table.

A point A dominates B if it is at least as accurate and at least as small, and strictly better in
one of the two. The Pareto frontier is the set of points nothing dominates: for a given amount of
silicon, the best accuracy achieved. The leaderboard is that frontier; there is no single winner
because best is a curve, not a number.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RECORDS = ROOT / "records"
RESULTS = ROOT / "results"

# categorical hues, fixed order, never cycled (see the dataviz palette)
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#7b5cd6"]
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#b9b8b4", "#fcfcfb"


def load_all(records: Path = RECORDS) -> list[dict]:
    out = []
    for res in sorted(records.glob("*/*/results.json")):
        r = json.loads(res.read_text())
        for p in r.get("points", []):
            out.append({**p, "record": r["record"], "title": r.get("title", r["record"])})
    return out


def frontier(points: list[dict]) -> list[dict]:
    """Points nothing dominates, sorted by area."""
    front = [
        p
        for p in points
        if not any(
            q["ge"] <= p["ge"]
            and q["test_acc"] >= p["test_acc"]
            and (q["ge"], -q["test_acc"]) != (p["ge"], -p["test_acc"])
            for q in points
        )
    ]
    return sorted(front, key=lambda p: p["ge"])


def _new_ax(plt, title: str, ylabel: str):
    import matplotlib.ticker as ticker

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    # label 1/2/5 per decade in plain numbers -- "10^4" tells a reader nothing about a gate budget
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=30))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{v/1e9:g}B" if v >= 1e9 else f"{v/1e6:g}M" if v >= 1e6
        else f"{v/1e3:g}k" if v >= 1e3 else f"{v:g}"))
    ax.set_xlabel("circuit size  (gate equivalents, sky130)", color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=13, loc="left", pad=12)
    ax.grid(True, which="both", color=MUTED, alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    return fig, ax


def _save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}")


def _powerlaw(ge: np.ndarray, y: np.ndarray) -> "tuple[float, float]":
    """Fit y = A * GE^b in log-log; return (A, b). y must be positive (an error/loss, not accuracy)."""
    b, a = np.polyfit(np.log(ge), np.log(y), 1)
    return float(np.exp(a)), float(b)


def plot_accuracy(points: list[dict], out: Path, extrapolate_to: float = 1e9) -> None:
    """Accuracy vs gate equivalents. Accuracy itself saturates, so the trendline is fit on the
    ERROR (100 - acc), which is a power law, and mapped back -- the same fit as the loss plot,
    shown in accuracy units. Measured solid, extrapolation dashed past the largest real circuit."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = sorted({p["record"] for p in points})
    fig, ax = _new_ax(plt, "MNIST accuracy vs circuit size", "MNIST test accuracy  (%)")
    for i, rec in enumerate(records):
        ps = sorted([p for p in points if p["record"] == rec], key=lambda p: p["ge"])
        c = SERIES[i % len(SERIES)]
        ge = np.array([p["ge"] for p in ps], float)
        acc = np.array([p["test_acc"] for p in ps], float)
        ax.plot(ge, acc, color=c, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, zorder=3, label=rec)
        err = np.clip(100.0 - acc, 1e-3, None)  # power law lives in error space, not accuracy
        if len(ps) >= 2:
            A, b = _powerlaw(ge, err)
            xs = np.geomspace(ge[-1], extrapolate_to, 50)
            ax.plot(xs, 100.0 - A * xs**b, color=c, lw=1.6, ls=(0, (5, 3)), alpha=0.7, zorder=2)
    ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=INK2)
    _save(fig, out)


def plot_loss(points: list[dict], out: Path, extrapolate_to: float = 1e9) -> None:
    """Cross-entropy vs gate equivalents, log-log. Measured points are drawn solid; a power-law fit
    to each record's measured points is extended as a DASHED trendline into the region we cannot
    synthesise (past the largest measured circuit), and the plot says so."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    pts = [p for p in points if p.get("test_ce") is not None]
    if not pts:
        return
    records = sorted({p["record"] for p in pts})
    fig, ax = _new_ax(plt, "MNIST loss at a fixed silicon budget  (log-log)",
                      "MNIST test cross-entropy  (log)")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0), numticks=30))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))

    for i, rec in enumerate(records):
        ps = sorted([p for p in pts if p["record"] == rec], key=lambda p: p["ge"])
        c = SERIES[i % len(SERIES)]
        ge = np.array([p["ge"] for p in ps], float)
        ce = np.array([p["test_ce"] for p in ps], float)
        ax.plot(ge, ce, color=c, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, zorder=3, label=rec)
        # power law CE = A * GE^b is a line in log-log; the dashed part extends it past the last
        # measured point, so solid = measured and dashed = extrapolated.
        if len(ps) >= 2:
            A, b = _powerlaw(ge, ce)
            xs = np.geomspace(ge[-1], extrapolate_to, 50)
            ax.plot(xs, A * xs**b, color=c, lw=1.6, ls=(0, (5, 3)), alpha=0.7, zorder=2)
    ax.legend(frameon=False, loc="lower left", fontsize=9, labelcolor=INK2)
    _save(fig, out)


def table(points: list[dict]) -> str:
    front = {(p["record"], p["name"]) for p in frontier(points)}
    rows = sorted(points, key=lambda p: (-p["test_acc"], p["ge"]))
    lines = [
        "| | record | point | gate equivalents | depth | MNIST test acc | test CE |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in rows:
        star = "*" if (p["record"], p["name"]) in front else ""
        ce = f"{p['test_ce']:.3f}" if p.get("test_ce") is not None else "--"
        lines.append(
            f"| {star} | `{p['record']}` | {p['name']} | {p['ge']:,.0f} "
            f"| {p['depth']} | **{p['test_acc']:.2f}%** | {ce} |"
        )
    lines.append("")
    lines.append("`*` = on the Pareto frontier (nothing is both smaller and more accurate). "
                 "test CE = calibrated cross-entropy over the circuit's class votes.")
    return "\n".join(lines)


def main() -> None:
    points = load_all()
    if not points:
        raise SystemExit("no results yet -- run `python -m mnistbench run records/<user>/<method>`")
    RESULTS.mkdir(exist_ok=True)
    plot_accuracy(points, RESULTS / "pareto_acc.png")
    plot_loss(points, RESULTS / "pareto_loss.png")
    md = table(points)
    (RESULTS / "leaderboard.md").write_text(md + "\n")

    readme = ROOT / "README.md"
    if readme.exists():
        txt = readme.read_text()
        a, b = "<!-- LEADERBOARD -->", "<!-- /LEADERBOARD -->"
        if a in txt and b in txt:
            head, rest = txt.split(a, 1)
            _, tail = rest.split(b, 1)
            readme.write_text(f"{head}{a}\n{md}\n{b}{tail}")
            print("updated README leaderboard")
    print("\n" + md)
