"""Synthesis: the x-axis (silicon area) and the netlist we measure accuracy on.

Two yosys+ABC runs over the same Verilog, with the same logic-optimization script:

  1. `abc -liberty sky130` -> standard cells -> chip area in um^2. Divided by the area of a
     sky130 NAND2 (3.7536 um^2) this is the classic **gate equivalent (GE)**, the unit ASIC
     people actually use. It is the benchmark's x-axis. GE, not a raw NAND count, because a
     raw count says an XOR and an inverter both cost "1 gate" while in silicon one is 5x the
     other -- an architecture that leans on wide gates would look artificially cheap.

  2. `abc -g NAND` -> a NAND-only netlist as yosys JSON. Same boolean function, but a form we
     can simulate exactly (see netlist.py), which is where the y-axis (accuracy) comes from.
     Accuracy comes from running the circuit, not from self-reporting.

Both runs start from the entrant's Verilog, so the optimizer folds away anything the submission
wasted: dead gates, constant-driven logic, a pixel nobody reads. You are charged for the circuit
you need, not the one you wrote.

ABC gotcha: `resyn2` and friends are aliases from abc.rc, which yosys does not load. Passing one
makes ABC abort, and yosys still prints a stat block, so an empty design can read as a large
"compression". The scripts below are expanded to primitives, and _check() hard-fails on ABC
errors.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Point these at your own install; the defaults are "whatever is on PATH" and the sky130 liberty
# that ships with `conda install -c litex-hub open_pdks.sky130a`. See docs/RULES.md.
_EDA = Path(os.environ.get("MNISTBENCH_EDA", "/itet-stor/sbuehrer/net_scratch/conda_envs/eda"))
_LIB = "share/pdk/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"

# NB the conda install is preferred over PATH on purpose: this cluster's /usr/sepp/bin/yosys is a
# wrapper around an AlmaLinux build that dies on Debian with `libreadline.so.7: not found`.
_CONDA_YOSYS = _EDA / "bin/yosys"
YOSYS = (
    os.environ.get("MNISTBENCH_YOSYS")
    or (str(_CONDA_YOSYS) if _CONDA_YOSYS.exists() else None)
    or shutil.which("yosys")
    or "yosys"
)
LIBERTY = os.environ.get("MNISTBENCH_LIBERTY", str(_EDA / _LIB))

# Area of sky130_fd_sc_hd__nand2_1, the unit of the GE axis.
NAND2_AREA_UM2 = 3.7536

# The body of ABC's `resyn2` alias, written out (commas become spaces inside yosys's +script form).
_RESYN2 = "balance;rewrite;refactor;balance;rewrite;rewrite,-z;balance;refactor,-z;rewrite,-z;balance"

# One optimization effort for everyone. Ends in `map`, which targets whatever ABC was given:
# the liberty cells in run 1, NAND2/INV in run 2.
OPT = f"strash;{_RESYN2};dc2;{_RESYN2};resub,-K,8;dc2;{_RESYN2};map"


@dataclass
class Area:
    """The x-axis."""

    ge: float  # gate equivalents = area / area(NAND2)
    area_um2: float
    cells: int
    by_type: dict[str, int]


@dataclass
class Nand:
    """A NAND-only netlist: yosys JSON plus its cell counts."""

    netlist: dict
    nand: int
    inv: int

    @property
    def gates(self) -> int:
        return self.nand + self.inv


def _run(cmds: str, cwd: str, timeout: int) -> str:
    p = subprocess.run(
        [YOSYS, "-p", cmds], capture_output=True, text=True, cwd=cwd, timeout=timeout
    )
    log = p.stdout + p.stderr
    if "ABC script did not complete" in log or "cmd error" in log:
        err = "\n".join(l for l in log.splitlines() if "cmd error" in l)
        raise RuntimeError(f"ABC aborted -- any numbers from this run would be garbage:\n{err}")
    if p.returncode != 0:
        raise RuntimeError(f"yosys exit {p.returncode}:\n{log[-3000:]}")
    return log


def synth_area(sv: Path, top: str = "top", timeout: int = 14400) -> Area:
    """Map to sky130 standard cells and measure area -> gate equivalents."""
    cmds = (
        f"read_verilog -sv {sv.resolve()}; synth -top {top} -flatten -noabc; opt -full; "
        f"abc -liberty {LIBERTY} -script +{OPT}; opt_clean; stat -liberty {LIBERTY}"
    )
    with tempfile.TemporaryDirectory() as td:
        log = _run(cmds, td, timeout)

    tail = log[log.rfind("Printing statistics") :]
    m = re.search(r"Chip area for (?:top )?module '\\?" + re.escape(top) + r"':\s*([\d.]+)", tail)
    if not m:
        raise RuntimeError(f"no chip area in yosys stat output:\n{tail[-2000:]}")
    area = float(m.group(1))

    # stat -liberty prints "<count> <area> <name>", one line per cell type plus a "cells" total.
    # The area column may be in scientific notation (3.2E+03), so match it loosely.
    rows = [(int(n), name) for n, name in re.findall(r"^\s+(\d+)\s+\S+\s+(\S+)\s*$", tail, re.M)]
    total = next((n for n, name in rows if name == "cells"), None)
    by_type = {name: n for n, name in rows if name.startswith("sky130_")}

    unmapped = [name for _, name in rows if name.startswith("$")]
    if unmapped:
        raise RuntimeError(f"netlist is not fully mapped to standard cells: {unmapped}")
    # if these disagree, the stat block was parsed incompletely and every number here is suspect
    if total is None or total != sum(by_type.values()):
        raise RuntimeError(
            f"parsed {sum(by_type.values())} cells but yosys reports {total}:\n{tail[-2000:]}"
        )
    return Area(ge=area / NAND2_AREA_UM2, area_um2=area, cells=total, by_type=by_type)


def synth_nand(sv: Path, top: str = "top", timeout: int = 14400) -> Nand:
    """Map to NAND2 + INV only and return the netlist, for exact simulation."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "netlist.json"
        cmds = (
            f"read_verilog -sv {sv.resolve()}; synth -top {top} -flatten -noabc; opt -full; "
            f"abc -g NAND -script +{OPT}; opt_clean; stat; write_json {out}"
        )
        log = _run(cmds, td, timeout)
        netlist = json.loads(out.read_text())

    tail = log[log.rfind("Printing statistics") :]
    counts: dict[str, int] = {}
    for n, cell in re.findall(r"^\s+(\d+)\s+\$_(\w+)_\s*$", tail, re.M):
        counts[cell] = counts.get(cell, 0) + int(n)
    stray = set(counts) - {"NAND", "NOT"}
    if stray:
        raise RuntimeError(f"netlist is not NAND-only, found {stray}")
    return Nand(netlist=netlist, nand=counts.get("NAND", 0), inv=counts.get("NOT", 0))
