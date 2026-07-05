"""Exactness tests for the HASH GATE substrate (IDEAS v3, stage 1).

1. LUT-corner equivalence: with c_k = 2**k the hash cell (sum c_k x_k) mod 2**K must equal
   the classic OR-shift K-bit address, bit for bit, on random data.
2. Full-lever exactness, both --gate corners: build a small deep Win, fire EVERY lever
   (tt, rewire, coef, step, sign, cls, rebuild, rs, share, split) repeatedly; after each
   call the incremental state must match a from-scratch pass exactly (score AND every
   stored output row) and hval must stay in sync -- same bar as --check.
"""
import sys, torch
sys.argv = ["x"]
import importlib.util
spec = importlib.util.spec_from_file_location("cd", "scratch/cd.py")
cd = importlib.util.module_from_spec(spec); spec.loader.exec_module(cd)

torch.manual_seed(0)
dev = "cpu"

# ---- 1. LUT corner == classic K-bit address --------------------------------------------
for K in (3, 4, 6, 8):
    win = cd.Win(ci=3 * 2, hw=8, chs=[10], hws=[8], fan_in=K, max_copies=64, device=dev,
                 init_deg=(0, 0, 0), init_loc=0, init_res=0.0, gate="lut")
    D = 200
    X = (torch.rand(win.N, D) < 0.5).to(torch.uint8)
    src = cd.pack_bits(X)
    flat = torch.randint(win.N, (50, K), dtype=torch.long)
    coef = win.coef[:50]
    got = win._cells(flat, coef, src, D)
    ref = torch.zeros((50, D), dtype=torch.int16)
    for i in range(K):
        ref |= cd.unpack_bits(src[flat[:, i]], D).to(torch.int16) << i
    assert torch.equal(got, ref), f"LUT corner mismatch at K={K}"
    print(f"K={K}: hash executor == classic LUT address on {50*D} cells  OK")

# ---- 2. every lever, both corners, exactness after each call ---------------------------
for gate in ("lut", "ternary"):
    torch.manual_seed(1)
    win = cd.Win(ci=3 * 4, hw=32, chs=[20, 10, 10], hws=[32, 32, 32], fan_in=6,
                 max_copies=1024, device=dev, init_deg=(0, 0, 0), init_loc=2,
                 init_res=0.5, gate=gate)
    win.cap = 1 << 30
    D = 400
    X = (torch.rand(win.N, D) < 0.5).to(torch.uint8)
    y = torch.randint(0, cd.CLS, (D,))
    win.set_train(X, y, rows=512)
    if gate == "ternary":                       # signed step tables really are thresholds
        assert win.coef.min() >= 0 and win.coef.max() < win.M

    acc = dict.fromkeys(("tt", "cn", "cf", "st", "sg", "cl", "rb", "rs", "sh", "sp"), 0)
    for it in range(8):
        alive = win.alive.nonzero().flatten()
        l = it % win.L                          # _commit contract: one layer per package
        ids = alive[win._lay(alive) == l]
        seg = ids[torch.randperm(ids.numel())[:300]]
        acc["tt"] += win.tt_sweep(seg)
        acc["cn"] += win.rewire(seg, 8, 4, 0.5)
        acc["cf"] += win.coef_pass(seg, 8)
        acc["sg"] += win.sign_pass(seg)
        acc["cl"] += win.cls_pass(seg)
        acc["rb"] += win.rebuild_pass(seg)
        acc["rs"] += win.rs_pass(seg, 0.5, 3, 4, 0.1, 0.0, True)
        for _ in range(20):
            g = int(alive[torch.randint(alive.numel(), (1,))])
            if bool(win.alive[g]):
                acc["sh"] += int(win.share_move(g, int(torch.randint(3, (1,))),
                                                bool(torch.randint(2, (1,)))))
                acc["sp"] += int(win.split_move(g, int(torch.randint(3, (1,)))))
        sh = win.alive.nonzero().flatten()
        acc["st"] += win.step_pass(sh[torch.randperm(sh.numel())[:300]])
        drift = win.check()
        hdrift = abs(win._hinge(win.score) - win.hval)
        assert drift == 0.0, f"[{gate}] EXACTNESS BROKEN at iter {it}: {drift}"
        assert hdrift < 1e-3, f"[{gate}] HVAL DESYNC at iter {it}: {hdrift}"
    total = sum(acc.values())
    zf = float((win.coef[win.alive] == 0).float().mean())
    print(f"gate={gate:7s}: all levers exact over 8 iters, accepts={acc} "
          f"hinge={win.hval / win.D:.4f} coef_zero={zf:.3f}  OK")
    assert acc["cf"] > 0, f"[{gate}] coef lever never accepted -- suspicious"

print("\nALL HASH-GATE TESTS PASSED")
