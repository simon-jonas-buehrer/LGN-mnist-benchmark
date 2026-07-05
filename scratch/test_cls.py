"""Isolated exactness test for the cls_pass (learned output-class) lever.
Builds a small deep Win, random train, fires cls_pass repeatedly, and after each
call asserts the incremental state matches a from-scratch pass (score AND every stored
output row) exactly, and that hval stays in sync. Mirrors what --check does."""
import sys, torch
sys.argv = ["x"]
import importlib.util
spec = importlib.util.spec_from_file_location("cd", "scratch/cd.py")
cd = importlib.util.module_from_spec(spec); spec.loader.exec_module(cd)

torch.manual_seed(0)
dev = "cpu"
# deep, unshared (deg000), fan-in 6 — the live recipe
win = cd.Win(ci=3 * 4, hw=32, chs=[20, 10, 10], hws=[32, 32, 32], fan_in=6,
             max_copies=1024, device=dev, init_deg=(0, 0, 0), init_loc=2, init_res=0.5)
win.cap = 1 << 30
D = 400
X = (torch.rand(win.N, D) < 0.5).to(torch.uint8)
y = torch.randint(0, cd.CLS, (D,))
win.set_train(X, y, rows=512)

alive = win.alive.nonzero().flatten()
total = 0
for it in range(12):
    seg = alive[torch.randperm(alive.numel())[:300]]
    n = win.cls_pass(seg)
    total += n
    drift = win.check()
    hdrift = abs(win._hinge(win.score) - win.hval)
    print(f"iter {it:2d}: cls moves={n:4d}  max|state diff|={drift:.6f}  hinge drift={hdrift:.6f}")
    assert drift == 0.0, f"EXACTNESS BROKEN: state diff {drift}"
    assert hdrift < 1e-3, f"HVAL DESYNC: {hdrift}"

# sanity: ocls actually changed away from the init channel%CLS assignment
init_cls = (win._cyx(torch.arange(win.S, device=dev))[1] % cd.CLS).to(torch.int8)
moved = int((win.ocls != init_cls).sum())
print(f"\nTOTAL accepted class moves={total}  slots whose class changed from init={moved}")
print("hinge (lower=better):", round(win.hval / win.D, 4))
print("EXACTNESS OK" if total >= 0 else "??")
