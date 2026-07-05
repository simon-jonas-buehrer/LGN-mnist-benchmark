"""Merge N trained cd.py checkpoints into ONE net (population -> individual).

Binary nets compose exactly: the head is a sum of slot votes, so concatenating the
models' channels per layer yields a single net whose scores are the SUM of the parents'
scores (a vote ensemble), bit-for-bit -- and CD can keep training the union afterwards
(cross-model rewiring becomes possible: gates may tap the sibling's features).

Requires: same layer count, same spatial sizes, same num-bits/in-feats encoding, and
fully unshared gates (deg == 0 everywhere; step/copies unsupported by the remap).

    .venv/bin/python scratch/merge.py out.pt a.pt b.pt [c.pt ...]
"""
import sys
from pathlib import Path

import torch

OUT, *SRC = sys.argv[1:]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as _A  # noqa: E402  (Win builder; clobbers argv, so OUT/SRC read first)


def materialize(path):
    """Load a ckpt and split every shared gate down to one gate per slot (split_move is
    exactly output-neutral, so this changes nothing), returning slot-aligned arrays."""
    win, a, rnd = _A.build(path)
    sh = (win.deg.long().sum(1) > 0) & win.alive
    for g in sh.nonzero().flatten().tolist():
        while int(win.deg[g].long().sum()):
            d = int((win.deg[g] > 0).nonzero()[0])
            assert win.split_move(g, d), f"split failed on gate {g}"
    assert int(win.deg[win.alive].abs().max()) == 0
    ck = {k: getattr(win, k).cpu() for k in ("base", "conn", "tt", "coef", "msize",
                                             "deg", "step", "sgn", "ocls", "alive",
                                             "owner")}
    return ck | {"args": a, "round": rnd}


cks = [materialize(p) for p in SRC]
a0 = cks[0]["args"]
for ck in cks:
    a = ck["args"]
    assert a["spatial"] == a0["spatial"] and a["num_bits"] == a0["num_bits"]
    assert a.get("in_feats", "none") == a0.get("in_feats", "none")
    assert a["fan_in"] == a0["fan_in"]
    assert int(ck["deg"].abs().max()) == 0, "merge supports unshared gates only"

chs = [[int(v) for v in ck["args"]["channels"].split(",")] for ck in cks]
L = len(chs[0])
assert all(len(c) == L for c in chs)
hws = [int(v) for v in a0["spatial"].split(",")]
if len(hws) == 1:
    hws = hws * L
mch = [sum(c[l] for c in chs) for l in range(L)]                 # merged channels per layer
ci = (8 if a0.get("in_feats", "none") == "edge" else 3) * a0["num_bits"]
N = ci * 32 * 32
S = sum(c * h * h for c, h in zip(mch, hws))

def cums(cl):
    out = [0]
    for c, h in zip(cl, hws):
        out.append(out[-1] + c * h * h)
    return out

mcum = cums(mch)
coff = [[sum(chs[j][l] for j in range(m)) for l in range(L)] for m in range(len(cks))]

def remap_slots(m, slots, cum_m):
    """Model-m slot ids -> merged slot ids (channel-block concat per layer)."""
    out = torch.empty_like(slots)
    for l in range(L):
        h = hws[l]
        sel = (slots >= cum_m[l]) & (slots < cum_m[l + 1])
        loc = slots[sel] - cum_m[l]
        c, r = loc // (h * h), loc % (h * h)
        out[sel] = mcum[l] + (c + coff[m][l]) * h * h + r
    return out

new = {k: [] for k in ("base", "conn", "tt", "coef", "msize", "deg", "step", "sgn",
                       "ocls", "alive", "owner")}
order = torch.empty(0, dtype=torch.long)
for m, ck in enumerate(cks):
    cum_m = cums(chs[m])
    Sm = cum_m[-1]
    conn = ck["conn"].long()
    inp = conn < N
    conn = torch.where(inp, conn, N + remap_slots(m, (conn - N).clamp(min=0), cum_m))
    base = remap_slots(m, ck["base"].long(), cum_m)
    # gate arrays are slot-aligned (unshared): reorder into merged slot order later
    new["conn"].append(conn.to(torch.int32))
    new["base"].append(base.to(torch.int32))
    for k in ("tt", "coef", "msize", "deg", "step", "sgn", "ocls", "alive"):
        if k in ck:
            new[k].append(ck[k])
        elif k == "msize":
            new[k].append(torch.full((Sm,), ck["tt"].shape[1], dtype=torch.int16))
    order = torch.cat([order, base])

perm = order.argsort()                                            # merged-slot order
out = {}
for k in ("conn", "tt", "coef", "msize", "deg", "step", "sgn", "ocls", "alive"):
    out[k] = torch.cat(new[k])[perm]
out["base"] = torch.cat(new["base"])[perm]
own = torch.empty(S, dtype=torch.int32)
own[out["base"].long()] = torch.arange(S, dtype=torch.int32)
out["owner"] = own
args = dict(a0)
args["channels"] = ",".join(map(str, mch))
out |= {"round": max(ck["round"] for ck in cks), "args": args}
torch.save(out, OUT)
print(f"merged {len(cks)} nets -> {OUT}: channels={args['channels']} slots={S} "
      f"(resume with --channels {args['channels']} --resume --ckpt {OUT})")
