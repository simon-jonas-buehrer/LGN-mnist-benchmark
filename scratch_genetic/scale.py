"""The scaling sweep: how good is a NAND net on CIFAR-10 as a function of HOW MANY GATES it has,
and for a given gate budget, what SHAPE (width vs depth) and what MUTATION SIZE is best?

Three axes, all log:
  * gates      1e2 .. 1e8 in half-decade steps (13 budgets)
  * depth      1, 2, 4, 8, 16, 32     -- width follows: W = (budget - readout) / D, so every run at
               a budget spends the SAME gates, just shaped differently
  * mut size   1, 2, 4, 8, ... 128    -- endpoints rewired per mutant. This axis matters because the
               search picks gates UNIFORMLY from the whole pool: in a big net almost every gate is
               dead weight (no path to the thin readout), so a 1-wire mutation almost never changes
               the output at all. Bigger nets may need bigger steps -- that is what this measures.

BUDGETED GRID. The full 13 x 6 x 8 = 592 cells would take ~178 h on one GPU. To fit the whole thing
in 24 h on ONE GPU we keep the full ladder where runs are cheap and coarsen where they are expensive:

    gates <= 1e5    full depth ladder x full mut ladder      cap  2 min   (converges early anyway)
    3e5 .. 3e6      full depth ladder x mut {1,4,16,64}      cap  6 min
    >= 1e7          depth {1,4,16}    x mut {1,8,32,128}     cap 10 min

A mutation bigger than ~10% of the net just randomizes the child, so cells with m > gates/10 are
dropped (a 128-wire rewire of a 100-gate net is not a mutation, it is a new genome).

    python scratch_genetic/scale.py --list         # the grid, one line per array task
    python scratch_genetic/scale.py --task 17      # the nand_ga.py flags for array task 17
    python scratch_genetic/scale.py --plot         # read runs/scale/*.jsonl -> curves + table

Task order is BY BUDGET ASCENDING, so a %1 array does the cheap runs first and the curve fills in
left to right.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs" / "scale"

EXPS = [2.0 + 0.5 * i for i in range(13)]     # 1e2 .. 1e8, half-decade steps
DEPTHS = [1, 2, 4, 8, 16, 32]
MUTS = [1, 2, 4, 8, 16, 32, 64, 128]
MIN_W = 16                                    # skip shapes thinner than this
MAX_MUT_FRAC = 0.1                            # a mutant may not rewire more than 10% of the gates


def readout(budget: int) -> int:
    """Thin output layer R: ~5% of the budget, a multiple of 10 (one vote group per class), clamped
    to [10, 2560]. It counts against the budget -- at 100 gates a readout is not free."""
    return max(10, min(2560, int(round(0.05 * budget / 10)) * 10))


def cell_plan(budget: int) -> tuple[list[int], list[int], float]:
    """(depths, mut sizes, minutes) for a budget -- the coarsening that makes the sweep fit in 24h."""
    if budget <= 1e5:
        return DEPTHS, MUTS, 2.0
    if budget <= 3.2e6:
        return DEPTHS, [1, 4, 16, 64], 6.0
    return [1, 4, 16], [1, 8, 32, 128], 10.0


def grid() -> list[dict]:
    out = []
    for e in EXPS:
        budget = int(round(10 ** e))
        R = readout(budget)
        depths, muts, minutes = cell_plan(budget)
        for D in depths:
            W = (budget - R) // D
            if W < MIN_W:
                continue
            gates = W * D + R
            for m in muts:
                if m > MAX_MUT_FRAC * gates:            # a mutation that big is just a re-roll
                    continue
                out.append({"exp": e, "budget": budget, "W": W, "D": D, "R": R, "m": m,
                            "gates": gates, "minutes": minutes, "name": f"e{e:.1f}_d{D}_m{m}"})
    return out


def flags(t: dict) -> str:
    # nand_ga sizes the batch and picks the low-memory forward itself (plan_memory). Big nets only
    # need a slower save/eval cadence: a 1e8-gate genome is ~800MB on disk, so checkpointing every
    # 500 gens would spend more time writing to NFS than searching.
    big = t["gates"] > 1e7
    return (f"--width {t['W']} --depth {t['D']} --out-width {t['R']} --mut-count {t['m']} "
            f"--eval-every {500 if big else 200} --ckpt-every {2000 if big else 500} "
            f"--max-minutes {t['minutes']} --out {RUNS}/{t['name']}")


# ==========================================================================================
def plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = {}                                                   # (gates, D, m) -> best val/test
    for t in grid():
        j = RUNS / f"{t['name']}.jsonl"
        if not j.exists():
            continue
        recs = [json.loads(ln) for ln in j.read_text().splitlines() if ln.strip()]
        if not recs:
            continue
        last = recs[-1]
        res[(t["gates"], t["D"], t["m"])] = {
            "val": last["best_val"], "test": last["best_test"], "gens": last["gen"],
            "min": last["min"], "exp": t["exp"]}
    if not res:
        print(f"no runs found under {RUNS}")
        return

    # Per budget, the best cell over (depth, mut) = the capacity curve; and which (depth, mut) won.
    by_budget = {}
    for (g, d, m), r in res.items():
        by_budget.setdefault(r["exp"], []).append((r["test"], d, m, g))
    env = sorted((max(v)[3], max(v)[0], max(v)[1], max(v)[2]) for v in by_budget.values())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax1, ax2, ax3 = axes

    # (1) accuracy vs gates: one line per depth (best mut for that depth), plus the envelope
    cmap = plt.get_cmap("viridis")
    for i, D in enumerate(DEPTHS):
        pts = {}
        for (g, d, m), r in res.items():
            if d == D:
                pts[g] = max(pts.get(g, 0), r["test"])
        if pts:
            xs = sorted(pts)
            ax1.plot(xs, [pts[x] for x in xs], "o-", ms=4,
                     color=cmap(i / max(1, len(DEPTHS) - 1)), label=f"depth {D}")
    ax1.plot([e[0] for e in env], [e[1] for e in env], "k--", lw=2, label="best cell", zorder=5)
    ax1.axhline(10, color="grey", ls=":", lw=1)                # chance
    ax1.set_xscale("log")
    ax1.set_xlabel("NAND gates")
    ax1.set_ylabel("test accuracy (%)")
    ax1.set_title("CIFAR-10 accuracy vs gate count\n(each depth at its best mutation size)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # (2) the optimal shape and step size for each budget
    ax2.plot([e[0] for e in env], [e[2] for e in env], "o-", color="tab:blue", label="best depth")
    ax2.plot([e[0] for e in env], [e[3] for e in env], "s-", color="tab:red", label="best mut size")
    ax2.set_xscale("log")
    ax2.set_yscale("log", base=2)
    ax2.set_xlabel("NAND gates")
    ax2.set_ylabel("optimal value")
    ax2.set_title("optimal depth and mutation size vs gate count")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # (3) accuracy vs mutation size, one line per budget -- does a bigger net need a bigger step?
    cmap2 = plt.get_cmap("plasma")
    exps = sorted({r["exp"] for r in res.values()})
    for i, e in enumerate(exps):
        pts = {}
        for (g, d, m), r in res.items():
            if r["exp"] == e:
                pts[m] = max(pts.get(m, 0), r["test"])
        if len(pts) > 1:
            xs = sorted(pts)
            ax3.plot(xs, [pts[x] for x in xs], "o-", ms=4,
                     color=cmap2(i / max(1, len(exps) - 1)), label=f"1e{e:g}")
    ax3.set_xscale("log", base=2)
    ax3.set_xlabel("mutation size (endpoints rewired per mutant)")
    ax3.set_ylabel("test accuracy (%)")
    ax3.set_title("accuracy vs mutation size, per gate budget\n(best depth for that cell)")
    ax3.legend(fontsize=7, ncol=2, title="gates")
    ax3.grid(alpha=0.3)

    fig.tight_layout()
    png = RUNS / "scaling.png"
    fig.savefig(png, dpi=130)
    print(f"wrote {png}  ({len(res)}/{len(grid())} cells done)\n")

    print("best cell per gate budget:")
    print(f"{'gates':>12} {'depth':>6} {'width':>10} {'mut':>5} {'test%':>7} {'val%':>7}")
    for g, acc, d, m in env:
        val = max(r["val"] for (gg, dd, mm), r in res.items() if gg == g)
        print(f"{g:>12,} {d:>6} {g // d:>10,} {m:>5} {acc:>7.2f} {val:>7.2f}")


# ==========================================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true")
    p.add_argument("--task", type=int, help="print nand_ga.py flags for this array index")
    p.add_argument("--count", action="store_true", help="print number of tasks")
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()
    g = grid()

    if a.count:
        print(len(g))
    elif a.task is not None:
        print(flags(g[a.task]))
    elif a.plot:
        plot()
    else:                                                      # --list (default)
        print(f"{len(g)} tasks | depths {DEPTHS} | muts {MUTS} | min width {MIN_W}")
        print(f"{'idx':>4} {'budget':>12} {'D':>4} {'W':>10} {'mut':>5} {'gates':>12} {'min':>5}")
        for i, t in enumerate(g):
            print(f"{i:>4} {t['budget']:>12,} {t['D']:>4} {t['W']:>10,} {t['m']:>5} "
                  f"{t['gates']:>12,} {t['minutes']:>5.0f}")
        est = sum(t["minutes"] for t in g) / 60
        print(f"\nworst case (every run hits its cap): {est:.1f} h serial on one GPU")
        print("early stopping (--patience) returns the unused time, so expect well under that")
