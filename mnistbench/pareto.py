"""Collect every record's results.json into the Pareto curve and the leaderboard table.

A point A dominates B if it is at least as accurate AND at least as small, and strictly better
in one of the two. The Pareto frontier is what nothing dominates: the answer to "for this much
silicon, what is the best accuracy anyone has achieved". That is the leaderboard -- there is no
single winner, because "best" is a curve, not a number.
"""

from __future__ import annotations

import json
from pathlib import Path

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


def plot(points: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    front = {(p["record"], p["name"]) for p in frontier(points)}
    records = sorted({p["record"] for p in points})

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # the frontier itself: a staircase behind the series, in ink, not in a series colour
    fr = frontier(points)
    if len(fr) > 1:
        ax.step(
            [p["ge"] for p in fr], [p["test_acc"] for p in fr],
            where="post", color=MUTED, lw=6, alpha=0.55, zorder=1, solid_capstyle="round",
            label="Pareto frontier",
        )

    for i, rec in enumerate(records):
        ps = sorted([p for p in points if p["record"] == rec], key=lambda p: p["ge"])
        c = SERIES[i % len(SERIES)]
        ax.plot([p["ge"] for p in ps], [p["test_acc"] for p in ps],
                color=c, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, zorder=3, label=rec)
        on = [p for p in ps if (p["record"], p["name"]) in front]
        ax.plot([p["ge"] for p in on], [p["test_acc"] for p in on], ls="none", marker="o",
                ms=8, mfc=c, mec=INK, mew=1.6, zorder=4)

    ax.set_xscale("log")
    # label 1/2/5 per decade in plain numbers -- "10^4" tells a reader nothing about a gate budget
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=20))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}")
    )
    ax.set_xlabel("circuit size  (gate equivalents, sky130)", color=INK2, fontsize=10)
    ax.set_ylabel("MNIST test accuracy  (%)", color=INK2, fontsize=10)
    ax.set_title("MNIST at a fixed silicon budget", color=INK, fontsize=13, loc="left", pad=12)
    ax.grid(True, which="both", color=MUTED, alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=INK2)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}")


def table(points: list[dict]) -> str:
    front = {(p["record"], p["name"]) for p in frontier(points)}
    rows = sorted(points, key=lambda p: (-p["test_acc"], p["ge"]))
    lines = [
        "| | record | point | gate equivalents | area (um^2) | depth | MNIST test acc |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in rows:
        star = "*" if (p["record"], p["name"]) in front else ""
        lines.append(
            f"| {star} | `{p['record']}` | {p['name']} | {p['ge']:,.0f} | {p['area_um2']:,.0f} "
            f"| {p['depth']} | **{p['test_acc']:.2f}%** |"
        )
    lines.append("")
    lines.append("`*` = on the Pareto frontier (nothing is both smaller and more accurate).")
    return "\n".join(lines)


def main() -> None:
    points = load_all()
    if not points:
        raise SystemExit("no results yet -- run `python -m mnistbench run records/<user>/<method>`")
    RESULTS.mkdir(exist_ok=True)
    plot(points, RESULTS / "pareto.png")
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
