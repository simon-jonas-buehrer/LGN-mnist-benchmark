"""Render a NAND netlist as a picture, so you can watch the circuit shrink.

yosys's own `show` shells out to graphviz, which gives up somewhere around a few hundred nodes --
useless here, where a single int8 conv kernel is ~15,000 gates. So we read yosys's JSON netlist
ourselves, compute each gate's logic depth, and lay the DAG out as depth (x) vs index-within-depth
(y). Wires are drawn as thin translucent lines: where the circuit is dense the lines pile up and
the region goes dark, so the plot doubles as a density map.

The point of the picture is the comparison -- same function, two netlists, side by side.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

YOSYS = "/itet-stor/sbuehrer/net_scratch/conda_envs/eda/bin/yosys"


def netlist_json(sv: Path, top: str, script: str) -> dict:
    """Synthesize to NAND-only and hand back yosys's JSON netlist."""
    sv = sv.resolve()  # yosys runs in a scratch cwd, so relative paths would not resolve
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "net.json"
        cmds = (
            f"read_verilog -sv {sv}; synth -top {top} -noabc; opt -full; "
            f"abc -g NAND -script +{script}; opt_clean; write_json {out}"
        )
        p = subprocess.run([YOSYS, "-p", cmds], capture_output=True, text=True, cwd=td, timeout=7200)
        if "cmd error" in p.stdout or "ABC script did not complete" in p.stdout:
            raise RuntimeError("ABC aborted; netlist is garbage")
        if not out.exists():
            raise RuntimeError(f"yosys produced no netlist:\n{p.stdout[-2000:]}\n{p.stderr[-1000:]}")
        return json.loads(out.read_text())


def build_dag(nl: dict, top: str) -> tuple[dict[int, list[int]], list[int], list[int]]:
    """Return driver-of-net, gate list, and each gate's input nets."""
    mod = nl["modules"][top]
    driver: dict[int, int] = {}  # net bit -> gate id that drives it
    gates: list[dict] = []
    for _, cell in mod["cells"].items():
        ins, outs = [], []
        for port, bits in cell["connections"].items():
            direction = cell["port_directions"][port]
            (outs if direction == "output" else ins).extend(b for b in bits if isinstance(b, int))
        gid = len(gates)
        gates.append({"in": ins, "out": outs, "type": cell["type"]})
        for b in outs:
            driver[b] = gid
    return driver, gates, []


def levels(driver: dict[int, int], gates: list[dict]) -> list[int]:
    """Longest-path logic depth of every gate (primary inputs are depth 0)."""
    depth = [-1] * len(gates)

    def d(g: int) -> int:
        if depth[g] >= 0:
            return depth[g]
        depth[g] = 0  # break combinational cycles defensively; there should be none
        best = 0
        for b in gates[g]["in"]:
            src = driver.get(b)
            if src is not None and src != g:
                best = max(best, d(src) + 1)
        depth[g] = best
        return best

    import sys

    sys.setrecursionlimit(200_000)
    for g in range(len(gates)):
        d(g)
    return depth


def render(sv: Path, top: str, scripts: dict[str, str], out_png: Path, title: str = "") -> None:
    fig, axes = plt.subplots(1, len(scripts), figsize=(7 * len(scripts), 7), facecolor="white")
    if len(scripts) == 1:
        axes = [axes]

    for ax, (label, script) in zip(axes, scripts.items()):
        nl = netlist_json(sv, top, script)
        driver, gates, _ = build_dag(nl, top)
        depth = levels(driver, gates)

        # position: x = logic depth, y = index within that depth
        seen: dict[int, int] = {}
        pos = []
        for g in range(len(gates)):
            dd = depth[g]
            i = seen.get(dd, 0)
            seen[dd] = i + 1
            pos.append((dd, i))
        # centre each column so the shape reads as a circuit, not a staircase
        colsize = dict(seen)
        pos = [(x, y - colsize[x] / 2) for x, y in pos]

        segs = []
        for g, gate in enumerate(gates):
            for b in gate["in"]:
                src = driver.get(b)
                if src is not None:
                    segs.append([pos[src], pos[g]])

        ax.add_collection(
            LineCollection(segs, linewidths=0.12, colors="#1f4e79", alpha=0.22, zorder=1)
        )
        xs = [p[0] for p in pos]
        ys = [p[1] for p in pos]
        ax.scatter(xs, ys, s=1.4, c="#c0392b", zorder=2, linewidths=0)

        ax.set_title(f"{label}\n{len(gates):,} gates   depth {max(depth)+1}", fontsize=13)
        ax.set_xlabel("logic depth")
        ax.autoscale()
        ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote {out_png}")
