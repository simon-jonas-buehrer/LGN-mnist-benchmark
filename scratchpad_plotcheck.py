import sys, numpy as np; sys.path.insert(0, ".")
from pathlib import Path
from mnistbench import pareto
pts = pareto.load_all()
# inject a plausible calibrated CE so we can validate the loss plot + trendline rendering
for p in pts:
    p["test_ce"] = round(float(3.5 * p["ge"]**-0.13), 4)  # fake power law, decreasing
pareto.plot_accuracy(pts, Path("scratchpad_acc.png"))
pareto.plot_loss(pts, Path("scratchpad_loss.png"))
print("OK", len(pts), "points")
