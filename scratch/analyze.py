"""Deep diagnostics on trained cd.py checkpoints: WHY does val stall?

    .venv/bin/python scratch/analyze.py scratch/gen5_anneal30.pt scratch/gen3_ctrl30.pt

Per model: per-class accuracy + confusions, margin anatomy (train-view vs val), layer
utilization (tap mix, cumulative/leave-one-out layer scores), gate health (dead rate,
output-rate distribution), functional redundancy (within-layer output correlations).
With two models: error overlap, oracle + vote-sum ensemble headroom.
"""
import sys
from pathlib import Path

import torch

PATHS = sys.argv[1:]
sys.argv = [sys.argv[0]]
import importlib.util

spec = importlib.util.spec_from_file_location("cd", Path(__file__).parent / "cd.py")
cd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cd)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Thermometer  # noqa: E402
from train import load_cifar10  # noqa: E402

CLS = 10
NAMES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def build(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    chs = [int(v) for v in a["channels"].split(",")]
    hws = [int(v) for v in a["spatial"].split(",")]
    if len(hws) == 1:
        hws = hws * len(chs)
    ci = (8 if a.get("in_feats", "none") == "edge" else 3) * a["num_bits"]
    win = cd.Win(ci, 32, chs, hws, a["fan_in"], a["max_copies"], "cpu",
                 init_deg=tuple(int(v) for v in a["init_deg"].split(",")),
                 init_loc=a["init_loc"], init_res=a["init_res"],
                 gate=a.get("gate", "lut"), tsize=a.get("tsize", 0),
                 margin=a.get("margin", 1.0), init_tsize=a.get("init_tsize", 0))
    for k in ("base", "conn", "tt", "coef", "msize", "deg", "alive", "owner", "step",
              "sgn", "ocls"):
        if k in ck:
            getattr(win, k).copy_(ck[k])
    return win, a, ck["round"]


def encoded(a):
    tx, ty, ex, ey = load_cifar10(Path(a["data_dir"]), False)
    nv = max(1, round(len(tx) * 0.1))
    vx, vy, px, py = tx[-nv:], ty[-nv:], tx[:-nv], ty[:-nv]
    feats = cd.in_feats if a.get("in_feats", "none") == "edge" else (lambda t: t)
    enc = Thermometer(num_bits=a["num_bits"]).fit(feats(px[:2000]))

    def encode(images):
        return enc(feats(images)).flatten(1).t().contiguous().to(torch.uint8)

    return encode, (px, py), (vx, vy), (ex, ey)


@torch.no_grad()
def forward_full(win, X):
    """Forward an encoded view; returns (score (CLS,D), slot output rates (S,), src)."""
    d = X.shape[1]
    pk = cd.pack_bits(X)
    src = torch.cat([pk, torch.zeros((win.S, pk.shape[1]), dtype=torch.int64)])
    score = win.forward(src, d, 2048)
    ones = torch.zeros(win.S)
    for s0 in range(0, win.S, 4096):
        ones[s0:s0 + 4096] = cd.unpack_bits(src[win.N + s0:win.N + s0 + 4096], d) \
            .float().sum(1)
    return score, ones / d, src


@torch.no_grad()
def analyze(path):
    win, a, rnd = build(path)
    encode, (px, py), (vx, vy), (ex, ey) = encoded(a)
    Xv = encode(vx)
    score, rate, src = forward_full(win, Xv)
    pred = score.argmax(0)
    acc = 100.0 * (pred == vy).float().mean().item()
    print(f"\n===== {path} (round {rnd})  val={acc:.2f}")

    # -- per-class accuracy and top confusions
    per = [(100.0 * ((pred == c) & (vy == c)).sum() / max(1, (vy == c).sum())).item()
           for c in range(CLS)]
    print("per-class val:", " ".join(f"{NAMES[c]}={per[c]:.0f}" for c in range(CLS)))
    conf = torch.zeros(CLS, CLS)
    for t, p in zip(vy.tolist(), pred.tolist()):
        conf[t, p] += 1
    conf.fill_diagonal_(0)
    top = conf.flatten().argsort(descending=True)[:5]
    print("top confusions:", ", ".join(
        f"{NAMES[i // CLS]}->{NAMES[i % CLS]} {int(conf[i // CLS, i % CLS])}"
        for i in map(int, top)))

    # -- margin anatomy on the val set (tau units)
    L = score / win.tau
    sy = L[vy, torch.arange(len(vy))]
    L2 = L.clone(); L2[vy, torch.arange(len(vy))] = -1e9
    marg = sy - L2.max(0).values
    q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    print(f"val margin (tau units): mean={marg.mean():.2f} "
          f"q={[round(float(v), 2) for v in torch.quantile(marg, q)]} "
          f"| wrong-and-confident (margin<-1): {100.0 * (marg < -1).float().mean():.1f}%")

    # -- layer utilization: tap source mix + cumulative & leave-one-out scores
    ids = win.alive.nonzero().flatten()
    lay = win._lay(ids)
    print("tap source mix per layer (reads input | reads each lower layer):")
    for l in range(win.L):
        taps = win.conn[ids[lay == l]].long().flatten()
        grid = torch.searchsorted(win.src_bound, taps, right=True)
        mix = torch.bincount(grid, minlength=win.L + 1).float()
        mix = (100 * mix / mix.sum()).tolist()
        print(f"  L{l}: " + " ".join(f"{v:.0f}" for v in mix[:l + 1]))
    slot_cls = win.ocls.long()
    slot_sgn = win.sgn.float()
    slot_lay = win._cyx(torch.arange(win.S))[0]
    votes = torch.zeros(win.L, CLS, Xv.shape[1])
    for s0 in range(0, win.S, 4096):
        out = cd.unpack_bits(src[win.N + s0:win.N + s0 + 4096], Xv.shape[1]).float()
        out *= slot_sgn[s0:s0 + 4096, None]
        for l in range(win.L):
            m = slot_lay[s0:s0 + 4096] == l
            if int(m.sum()):
                votes[l].index_add_(0, slot_cls[s0:s0 + 4096][m], out[m])
    cum = torch.zeros(CLS, Xv.shape[1])
    for l in range(win.L):
        cum += votes[l]
        a_cum = 100.0 * (cum.argmax(0) == vy).float().mean()
        a_loo = 100.0 * ((score - votes[l]).argmax(0) == vy).float().mean()
        print(f"  L{l}: cum-acc(<=L{l})={a_cum:.1f}  leave-out-acc={a_loo:.1f}"
              f"  ({int((lay == l).sum())} gates)")

    # -- gate health + functional redundancy
    dead = 100.0 * ((rate < 0.01) | (rate > 0.99)).float().mean()
    print(f"gate health: dead/constant={dead:.1f}%  rate "
          f"q={[round(float(v), 2) for v in torch.quantile(rate, q)]}")
    for l in range(win.L):
        sl = (slot_lay == l).nonzero().flatten()
        pick = sl[torch.randperm(sl.numel())[:400]]
        out = cd.unpack_bits(src[win.N + pick], Xv.shape[1]).float()
        out = out - out.mean(1, keepdim=True)
        sd = out.std(1) + 1e-6
        C = (out @ out.t()) / (sd[:, None] * sd[None] * Xv.shape[1])
        off = C[~torch.eye(len(pick), dtype=torch.bool)]
        print(f"  L{l} functional redundancy: mean|corr|={off.abs().mean():.3f} "
              f"p95|corr|={torch.quantile(off.abs(), torch.tensor(0.95)):.3f}")
    return score, vy


if __name__ == "__main__":
    scores = []
    for p in PATHS:
        scores.append(analyze(p))
    if len(scores) >= 2:
        (s1, y), (s2, _) = scores[0], scores[1]
        e1, e2 = s1.argmax(0) != y, s2.argmax(0) != y
        both = (e1 & e2).float().mean()
        print(f"\n===== cross-model error structure")
        print(f"err A={100 * e1.float().mean():.1f}%  err B={100 * e2.float().mean():.1f}%  "
              f"both-wrong={100 * both:.1f}%  oracle-ensemble={100 * (1 - both):.1f}")
        ens = 100.0 * ((s1 / s1.std() + s2 / s2.std()).argmax(0) == y).float().mean()
        print(f"vote-sum 2-ensemble val={ens:.2f}")
