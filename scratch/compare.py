"""Round-aligned comparison of runs: val/train/hinge per round for every matching jsonl.

    .venv/bin/python scratch/compare.py hg2 reg1 hg1_ctrl   # prefixes or full names
    .venv/bin/python scratch/compare.py                     # everything in scratch/
"""
import json
import sys
from pathlib import Path

pats = sys.argv[1:] or [""]
runs = {}
for f in sorted(Path(__file__).parent.glob("*.jsonl")):
    if not any(f.stem.startswith(p) or p in f.stem for p in pats):
        continue
    rec = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    rec = [r for r in rec if r.get("round", 0) > 0]
    if rec:
        runs[f.stem] = {r["round"]: r for r in rec}
if not runs:
    sys.exit("no matching jsonl runs")

rmax = max(max(rs) for rs in runs.values())
names = list(runs)
w = max(len(n) for n in names)
print(f"{'':>{w}} | " + " | ".join(f"r{r:<11d}" for r in range(1, rmax + 1)))
for n in names:
    cells = []
    for r in range(1, rmax + 1):
        rec = runs[n].get(r)
        cells.append(f"{rec['val'] or float('nan'):5.1f}/{rec['train']:5.1f}"
                     if rec else " " * 11)
    print(f"{n:>{w}} | " + " | ".join(cells))
print(f"{'(val/train)':>{w}}")

# per-run summary: best val, last round, minutes, key op accepts at last round
print()
for n in names:
    rs = runs[n]
    last = rs[max(rs)]
    best = max((r.get("val") or 0, rd) for rd, r in rs.items())
    line = (f"{n:>{w}} | best val {best[0]:5.2f} @r{best[1]:<3d} | last r{max(rs):<3d} "
            f"train {last['train']:5.2f} hinge {last['hinge']:.3f} "
            f"gap {last['train'] - (last.get('val') or float('nan')):5.1f} "
            f"| {last.get('min', 0):6.1f}min")
    for k in ("ttbits", "rewires", "coefs", "rs"):
        if k in last:
            line += f" {k[:4]}={last[k]}"
    if "coef_zero" in last:
        line += f" c0={last['coef_zero']:.3f}"
    print(line)
