"""Summarise the accuracy study (exp3.sbatch): best validation accuracy per config, sorted."""
import json
from pathlib import Path

rows = []
for f in sorted(Path("scratch_genetic/runs").glob("acc_*.jsonl")):
    recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    if not recs:
        continue
    bv = max(r["val"] for r in recs)
    bt = max(r["test"] for r in recs)
    last = recs[-1]
    rows.append((bv, bt, last["val"], last["gen"], last.get("gps", 0), f.stem[4:]))
rows.sort(reverse=True)
print(f"{'config':14s} {'best_val':>8} {'best_test':>9} {'final_val':>9} {'gens':>7} {'gen/s':>6}")
for bv, bt, fv, gen, gps, name in rows:
    print(f"{name:14s} {bv:8.2f} {bt:9.2f} {fv:9.2f} {gen:7d} {gps:6.1f}")
