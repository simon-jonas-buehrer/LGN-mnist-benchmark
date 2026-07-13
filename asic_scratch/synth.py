"""Synthesize a SystemVerilog module to a NAND-only netlist and count gates at each stage.

The pipeline turns a conv kernel into gates in three steps, and this file measures all three so
the shrinkage is attributable:

    generic   weights are runtime INPUTS  -> real int8 multipliers get instantiated
    folded    weights are CONSTANTS       -> multipliers collapse into a shift-add tree
    folded+opt  ABC minimizes the result  -> rewriting / refactoring / resubstitution

`generic -> folded` is "bake the weights into the gates"; `folded -> folded+opt` is logic
minimization proper. Both preserve the boolean function exactly, so `cec()` can prove it.

Gates are counted as NAND2 + INV because a NAND-only netlist is the target representation.

ABC gotcha, learned the hard way: `resyn2`, `resyn2rs` and friends are ALIASES defined in
abc.rc, which yosys does not load. Passing them makes ABC abort -- and yosys will still happily
print a stat block for the wreckage (an empty netlist reads as a spectacular 176x "compression").
So the scripts below are expanded to primitive commands, and _check() hard-fails on ABC errors.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

YOSYS = "/itet-stor/sbuehrer/net_scratch/conda_envs/eda/bin/yosys"

# The body of ABC's `resyn2` alias, written out. Commas become spaces inside yosys's +script form.
_RESYN2 = "balance;rewrite;refactor;balance;rewrite;rewrite,-z;balance;refactor,-z;rewrite,-z;balance"

BASELINE = "strash;map"
OPTIMIZED = f"strash;{_RESYN2};dc2;{_RESYN2};resub,-K,8;dc2;{_RESYN2};map"


@dataclass
class Result:
    nand: int
    inv: int

    @property
    def gates(self) -> int:
        return self.nand + self.inv


def _check(log: str) -> None:
    if "ABC script did not complete" in log or "cmd error" in log:
        err = "\n".join(l for l in log.splitlines() if "cmd error" in l)
        raise RuntimeError(f"ABC aborted -- gate counts would be garbage:\n{err}")
    if "ERROR" in log:
        raise RuntimeError(f"yosys error:\n{log[-2000:]}")


def _parse(log: str) -> Result:
    tail = log[log.rfind("Printing statistics") :]
    counts: dict[str, int] = {}
    for m in re.finditer(r"^\s+(\d+)\s+\$_(\w+)_\s*$", tail, re.M):
        counts[m.group(2)] = counts.get(m.group(2), 0) + int(m.group(1))
    stray = set(counts) - {"NAND", "NOT"}
    if stray:
        raise RuntimeError(f"netlist is not NAND-only, found {stray}")
    return Result(nand=counts.get("NAND", 0), inv=counts.get("NOT", 0))


def synth(sv: Path, top: str, script: str = OPTIMIZED, netlist_out: Path | None = None) -> Result:
    write = f"; write_verilog -noattr {netlist_out}" if netlist_out else ""
    cmds = (
        f"read_verilog -sv {sv}; synth -top {top} -noabc; opt -full; "
        f"abc -g NAND -script +{script}; opt_clean; stat{write}"
    )
    with tempfile.TemporaryDirectory() as td:
        p = subprocess.run([YOSYS, "-p", cmds], capture_output=True, text=True, cwd=td, timeout=7200)
    _check(p.stdout + p.stderr)
    if p.returncode != 0:
        raise RuntimeError(f"yosys exit {p.returncode}:\n{p.stdout[-2000:]}")
    return _parse(p.stdout)


def cec(sv_a: Path, top_a: str, sv_b: Path, top_b: str) -> bool:
    """Prove two designs compute the same boolean function (miter + SAT via yosys `equiv`)."""
    cmds = (
        f"read_verilog -sv {sv_a}; rename {top_a} gold; "
        f"read_verilog -sv {sv_b}; rename {top_b} gate; "
        "equiv_make gold gate miter; hierarchy -top miter; "
        "equiv_simple; equiv_status -assert"
    )
    with tempfile.TemporaryDirectory() as td:
        p = subprocess.run([YOSYS, "-p", cmds], capture_output=True, text=True, cwd=td, timeout=7200)
    return p.returncode == 0
