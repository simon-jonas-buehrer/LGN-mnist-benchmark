"""Vote-sum ensemble over trained checkpoints (binary nets: class scores just add)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as A  # noqa: E402  (reuses build/encoded/forward_full; clobbers argv)

import torch  # noqa: E402

scores, vy = [], None
for p in ["scratch/gen5_anneal30.pt", "scratch/gen3_ctrl30.pt", "scratch/gen5_long60.pt",
          "scratch/gen5_temp0.pt", "scratch/gen5_hcap3.pt", "scratch/gen3_jit6.pt"]:
    try:
        win, a, rnd = A.build(p)
        encode, _, (vx, vy), _ = A.encoded(a)
        s, _, _ = A.forward_full(win, encode(vx))
        acc = 100.0 * (s.argmax(0) == vy).float().mean().item()
        print(f"{p}: r{rnd} val={acc:.2f}", flush=True)
        scores.append(s / s.std())
    except Exception as e:
        print(f"{p}: skip ({type(e).__name__}: {e})", flush=True)
for k in range(2, len(scores) + 1):
    ens = sum(scores[:k])
    print(f"ensemble of {k}: val={100.0 * (ens.argmax(0) == vy).float().mean().item():.2f}",
          flush=True)
