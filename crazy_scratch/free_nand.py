"""crazy_scratch: a FREE-DAG NAND circuit trained by variable-size zeroth-order search.

Departures from scratch/opt.py's fixed monarch LUT net -- here we learn ONLY the wiring:

  * Every gate is a FIXED 2-input NAND. There is no truth table to learn; the ONLY
    parameters are the connections -- which two signals each gate reads.
  * A gate at depth l may read ANY signal at strictly lower depth (the encoded input bits,
    or any gate in layers 0..l-1). That is a free ACYCLIC wiring -- the strict depth order
    is what forbids cycles -- not a layer-local tap. Each of a gate's two taps carries
    k fixed random candidate sources drawn from that whole allowed range.
  * Functional completeness: a 2-input NAND is already universal on its own (NAND(a,a)=~a is
    NOT, and AND/OR/... build from there), so no constants are needed. The one thing worth
    guaranteeing is that NOT is always REACHABLE: ONE of tap 1's k candidates (slot 1) is not
    a fixed source but "the other input" -- it resolves to whatever tap 0 currently reads --
    so NAND(a,a)=~a of any signal tap 0 can select is always one selection away.
  * Candidate sources are UNIFORM/global by default (--local-sigma 0): each tap's k
    candidates are drawn uniformly over ALL earlier signals. --local-sigma > 0 instead biases
    them toward a recent layer (--depth-decay) at a nearby width position (spatial locality).
    MEASURED (2026-07-07, f=2 depth-16, 100M samples): locality is a ~2-8pt test-acc HANDICAP
    -- the free global wiring wins, strongly so when combined with CE + large n.
  * One unified move interpolates coordinate descent and random search: sample n from an
    exponential distribution (--n-mean), re-draw n random taps to random candidates AT ONCE,
    keep the change iff the batch loss strictly drops. n~1 is coordinate descent; larger n is
    random search -- moves touch more taps so they bite harder. The batch is held fixed for
    --props proposals so accepted changes compound.
  * Loss (--loss) is either softmax CROSS-ENTROPY on the group popcounts (default 'ce'; the
    opt.py GroupSum head, shift-invariant so it has no all-dark minimiser) or class-balanced
    per-bit BCE ('bce') of the (B, CLS*h) bits vs the one-hot-per-group target (correct group
    all 1, rest 0). On hard bits clamped BCE is affine in Hamming, so bce minimises Hamming
    directly; the target is h ones vs (CLS-1)*h zeros, so raw BCE collapses to the all-dark
    circuit and --pos-weight (default CLS-1) class-balances it. Accuracy is always argmax of
    the per-group popcounts.

    python crazy_scratch/free_nand.py --out crazy_scratch/runs/free0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Thermometer  # noqa: E402
from train import load_cifar10  # noqa: E402

CLS = 10
TOPO_SEED = 0  # candidate wiring: identical for every seed/run
MIRROR = 1     # tap 1's candidate slot 1 is not a fixed source -- it MIRRORS tap 0 (the
               # other input), so NAND(a, a) = ~a (NOT) is always one selection away


# ==========================================================================================
# Augmentation (copied from scratch/opt.py so this folder is self-contained)
# ==========================================================================================
def augment(x: torch.Tensor, crop: int = 4) -> torch.Tensor:
    """Light augmentation on (B, 3, 32, 32) in [0,1]: random h-flip + random crop-4."""
    d = x.shape[0]
    fl = torch.rand(d, device=x.device) < 0.5
    x = torch.where(fl[:, None, None, None], x.flip(-1), x)
    if crop:
        p = F.pad(x, (crop,) * 4, mode="replicate").permute(0, 2, 3, 1)
        oy = torch.randint(0, 2 * crop + 1, (d,), device=x.device)
        ox = torch.randint(0, 2 * crop + 1, (d,), device=x.device)
        b = torch.arange(d, device=x.device)[:, None, None]
        ys = (oy[:, None] + torch.arange(32, device=x.device))[:, :, None]
        xs = (ox[:, None] + torch.arange(32, device=x.device))[:, None, :]
        x = p[b, ys, xs].permute(0, 3, 1, 2)
    return x


# ==========================================================================================
# Topology: per-tap candidate sources over the whole earlier DAG (no constants)
# ==========================================================================================
def build_candidates(I: int, W: int, L: int, k: int, sigma: float, ddecay: float,
                     device: str) -> list[torch.Tensor]:
    """Per layer: (W, 2, k) int64 GLOBAL signal indices into the flat buffer

        S = [ inputs(I) | layer0(W) | layer1(W) | ... | layer{L-1}(W) ]

    so a gate at layer l may reference any column < I + l*W (an input or a shallower gate)
    -- acyclic by construction. Candidates come from a fixed generator (shared across seeds).
    Slot 0 is pinned to the gate directly below at the same width position, so the default
    all-slot-0 selection is NAND(below, below) = ~below, an inverter chain that propagates
    signal from step 0 (residual init). No constant wires -- NAND is already universal. Tap
    1's slot MIRROR resolves in sel_to_conn to whatever tap 0 reads, so a NOT is always
    reachable.

    LOCALITY (sigma > 0): each other candidate reads a source d layers back -- d = 1 + Exp
    with mean `ddecay`, so recent layers dominate (depth locality; d beyond l falls through to
    the inputs) -- at a width position drawn from a Gaussian of std `sigma` around the gate's
    own x, wrapped (spatial locality: adjacent x ~ adjacent encoder-grid positions). sigma <= 0
    falls back to the old uniform-over-all-earlier-signals draw (global random wiring)."""
    gen = torch.Generator().manual_seed(TOPO_SEED)
    j = torch.arange(W)
    cands = []
    for l in range(L):
        cand = torch.empty(W, 2, k, dtype=torch.int64)
        for s in range(k):
            if sigma > 0:
                d = 1 + torch.empty(W, 2).exponential_(1.0 / ddecay, generator=gen).long()
                lsrc = l - d                                 # source layer; < 0 => inputs
                off = (sigma * torch.randn(W, 2, generator=gen)).round().long()
                gidx = I + lsrc.clamp(min=0) * W + torch.remainder(j[:, None] + off, W)
                iidx = torch.remainder((j[:, None] % I) + off, I)
                cand[:, :, s] = torch.where(lsrc < 0, iidx, gidx)
            else:
                cand[:, :, s] = torch.randint(0, I + l * W, (W, 2), generator=gen)
        below = (I + (l - 1) * W + j) if l > 0 else (j % I)
        cand[:, 0, 0] = below
        cand[:, 1, 0] = below                                # both taps -> ~below at init
        cands.append(cand.to(device))
    return cands


def sel_to_conn(cand: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
    """(W, 2, k) candidates + (W, 2) selection -> (W, 2) concrete global sources. Tap 1's
    slot MIRROR is dynamic: it resolves to tap 0's chosen source, giving NAND(a, a) = ~a."""
    conn0 = cand[:, 0].gather(1, sel[:, 0:1]).squeeze(1)
    conn1 = cand[:, 1].gather(1, sel[:, 1:2]).squeeze(1)
    conn1 = torch.where(sel[:, 1] == MIRROR, conn0, conn1)
    return torch.stack((conn0, conn1), 1)


# ==========================================================================================
# Hard executor: one flat uint8 buffer, NAND all the way through
# ==========================================================================================
@torch.no_grad()
def fwd_into(S: torch.Tensor, conns: list[torch.Tensor], I: int, W: int,
             from_layer: int = 0) -> None:
    """Fill S in place from `from_layer` down. Layer l reads columns < I+l*W (already valid)
    and writes its block. Recompute after a connection change starts at the shallowest touched
    layer; columns before that block are untouched."""
    for l in range(from_layer, len(conns)):
        c = conns[l]
        s = I + l * W
        S[:, s:s + W] = (S[:, c[:, 0]] & S[:, c[:, 1]]) ^ 1  # NAND


def head(bits: torch.Tensor, y: torch.Tensor, mode: str,
         pw: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, W) final bits -> (per-sample loss, (B, CLS) group popcounts=logits). The head is
    GroupSum: logit c = popcount of the h bits of group c; accuracy is always argmax(g).

    mode 'ce': softmax cross-entropy on logits = popcount / sqrt(h) (the opt.py head). Being
    shift-invariant it has NO all-dark minimiser, so no class weighting is needed.
    mode 'bce': class-balanced per-bit BCE vs the one-hot-per-group target (correct group all
    1, rest 0). On hard bits clamped BCE is affine in Hamming = pw*(h - pop_y) + (total - pop_y)
    /used; pw=CLS-1 balances the ones against the (CLS-1)x more numerous zeros (pw=1 = raw BCE,
    which collapses to all-dark)."""
    B, Wl = bits.shape
    h = Wl // CLS                                            # bits per class group
    u = CLS * h                                              # used bits (drop remainder < CLS)
    g = bits[:, :u].view(B, CLS, h).sum(-1, dtype=torch.int32)  # per-group popcount = logits
    if mode == "ce":
        loss = F.cross_entropy(g.to(torch.float32) / math.sqrt(h), y, reduction="none")
    else:
        pop_y = g.gather(1, y[:, None]).squeeze(1).to(torch.float32)
        total = g.sum(-1).to(torch.float32)
        loss = (pw * (h - pop_y) + (total - pop_y)) / u
    return loss, g


# ==========================================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/cifar-10-batches-py"))
    p.add_argument("--download", action="store_true")
    p.add_argument("--num-bits", type=int, default=4, help="thermometer bits per channel")
    p.add_argument("--depth", type=int, default=16)
    p.add_argument("--f", type=int, default=2,
                   help="copy factor: width = C*b*H*W*f = I*f, i.e. f gates per encoder "
                        "output bit at init (each encoder bit is the residual 'below' of f "
                        "layer-0 gates)")
    p.add_argument("--width", type=int, default=0, help="explicit gates per layer; 0 = I*f")
    p.add_argument("--k", type=int, default=16, help="candidate sources per tap (tap 1's "
                   "slot 1 mirrors tap 0 -- the other input -- so NOT is always reachable)")
    p.add_argument("--crop", type=int, default=4)
    p.add_argument("--batch", type=int, default=4096, help="large: a batch win is a real win")
    p.add_argument("--props", type=int, default=64, help="proposals per batch re-roll")
    p.add_argument("--n-mean", type=float, default=4.0,
                   help="mean of the Exp added to 1 for the mutation size n (~1 => CD, larger "
                        "=> more RS: moves touch more taps at once so they bite harder)")
    p.add_argument("--n-max", type=int, default=256, help="cap on taps mutated per proposal")
    p.add_argument("--local-sigma", type=float, default=0.0,
                   help="0 = uniform-over-all-earlier-signals (global free-DAG wiring; the "
                        "empirical winner). >0 = width-position Gaussian std for spatial "
                        "locality -- MEASURED WORSE here (see module docstring / memory)")
    p.add_argument("--depth-decay", type=float, default=1.0,
                   help="mean layers-back for candidate sources (depth locality; small = read "
                        "mostly the layer directly below)")
    p.add_argument("--loss", choices=["ce", "bce"], default="ce",
                   help="ce: softmax cross-entropy on group popcounts (shift-invariant, no "
                        "all-dark collapse); bce: class-balanced per-bit BCE vs one-hot groups")
    p.add_argument("--pos-weight", type=float, default=CLS - 1,
                   help="bce only: class-balance weight on the correct group's bits (CLS-1 "
                        "balances ones vs zeros; 1 = raw per-bit BCE, collapses to all-dark)")
    p.add_argument("--max-samples", type=float, default=100e6)
    p.add_argument("--max-minutes", type=float, default=0.0)
    p.add_argument("--eval-mins", type=float, default=2.0)
    p.add_argument("--out", type=Path, required=True, help="prefix for .jsonl and .pt")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    dev = args.device
    torch.manual_seed(args.seed)

    # ---- data + encoder ------------------------------------------------------------------
    tx, ty, ex, ey = load_cifar10(args.data_dir, args.download)
    nv = 5000
    px, py = tx[:-nv].to(dev), ty[:-nv].to(dev)
    enc = Thermometer(num_bits=args.num_bits).fit(tx[:2000]).to(dev)
    I = 3 * args.num_bits * 32 * 32
    W = args.width or I * args.f                              # width = C*b*H*W*f = I*f
    L, k = args.depth, args.k
    pw = args.pos_weight
    T = I + L * W                                             # flat signal-buffer width

    def encode(images):
        return enc(images).flatten(1).to(torch.uint8)

    evals = {"train": (encode(px[:5000]), py[:5000]),
             "val": (encode(tx[-nv:].to(dev)), ty[-nv:].to(dev))}
    test = (ex, ey)

    # ---- parameters: candidates + per-tap selection (selection 0 everywhere = the residual
    #      inverter-chain init from build_candidates) ---------------------------------------
    cand = build_candidates(I, W, L, k, args.local_sigma, args.depth_decay, dev)
    sel = [torch.zeros(W, 2, dtype=torch.int64, device=dev) for _ in range(L)]
    conns = [sel_to_conn(c, s) for c, s in zip(cand, sel)]
    fin = I + (L - 1) * W                                     # start of the final block

    def run(S, cs, xb):
        S[:, :I] = xb
        fwd_into(S, cs, I, W, 0)
        return S[:, fin:]

    @torch.no_grad()
    def eval_sets(final=False):
        sets = dict(evals)
        if final:
            sets["test"] = (encode(test[0].to(dev)), test[1].to(dev))
        out = {}
        Se = torch.empty(2048, T, dtype=torch.uint8, device=dev)
        for name, (xe, ye) in sets.items():
            ls, co, nn = 0.0, 0, len(xe)
            for i in range(0, nn, 2048):
                xb, yb = xe[i:i + 2048], ye[i:i + 2048]
                bits = run(Se[:len(xb)], conns, xb)
                ham, g = head(bits, yb, args.loss, pw)
                ls += ham.sum().item()
                co += (g.argmax(1) == yb).sum().item()
            loss = ls / nn
            out[name] = {"loss": round(loss, 5), "acc": round(100.0 * co / nn, 2)}
        return out

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        out.with_suffix(".jsonl").write_text("")
    samples = 0.0
    t0 = time.time()
    last_eval = [None]
    accepts = trials = 0

    def due():
        now = time.time()
        if last_eval[0] is None or now - last_eval[0] >= args.eval_mins * 60:
            last_eval[0] = now
            return True
        return False

    def done():
        if samples >= args.max_samples:
            return True
        return bool(args.max_minutes) and (time.time() - t0) / 60 >= args.max_minutes

    def log(final=False):
        m = eval_sets(final=final)
        rec = {"samples": int(samples), "min": round((time.time() - t0) / 60, 1),
               "accepts": accepts, "trials": trials, **m}
        with open(out.with_suffix(".jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        torch.save({"args": vars(args) | {"out": str(out)}, "samples": int(samples),
                    "sel": [s.cpu() for s in sel]}, out.with_suffix(".pt"))
        te = f" test {m['test']['loss']:.4f}/{m['test']['acc']:5.2f}" if final else ""
        print(f"{int(samples):>12,} | train {m['train']['loss']:.4f}/{m['train']['acc']:5.2f}"
              f" | val {m['val']['loss']:.4f}/{m['val']['acc']:5.2f}{te} | "
              f"acc/trial {accepts}/{trials} | {rec['min']:6.1f}m", flush=True)

    print(f"free_nand I={I} f={args.f} width={W} depth={L} k={k} T={T} h={W // CLS} "
          f"batch={args.batch} props={args.props} n_mean={args.n_mean} loss={args.loss} "
          f"sigma={args.local_sigma} ddecay={args.depth_decay} pw={pw} "
          f"train={len(px)} val={nv}", flush=True)

    # ---- training buffer + loop ----------------------------------------------------------
    S = torch.empty(args.batch, T, dtype=torch.uint8, device=dev)
    while not done():
        if due():
            log()
        idx = torch.randint(len(px), (args.batch,), device=dev)
        xb = encode(augment(px[idx], args.crop))
        yb = py[idx]
        base = head(run(S, conns, xb), yb, args.loss, pw)[0].mean().item()
        samples += args.batch
        for _ in range(args.props):
            trials += 1
            u = torch.rand(()).item()
            n = min(args.n_max, 1 + int(-args.n_mean * math.log(1.0 - u)))
            flat = torch.randint(L * W * 2, (n,), device=dev)
            lz = (flat // (W * 2)).tolist()
            gz = ((flat % (W * 2)) // 2).tolist()
            tz = (flat % 2).tolist()
            nz = torch.randint(k, (n,), device=dev).tolist()
            olds = []
            touched = set()
            for li, gi, ti, ni in zip(lz, gz, tz, nz):
                olds.append((li, gi, ti, int(sel[li][gi, ti])))
                sel[li][gi, ti] = ni
                touched.add(li)
            for li in touched:
                conns[li] = sel_to_conn(cand[li], sel[li])
            fl = min(touched)
            start = I + fl * W
            old_suffix = S[:, start:].clone()
            fwd_into(S, conns, I, W, fl)
            samples += args.batch * (L - fl) / L
            cur = head(S[:, fin:], yb, args.loss, pw)[0].mean().item()
            if cur < base:
                base, accepts = cur, accepts + 1
            else:                                            # revert selection + buffer
                for li, gi, ti, ov in olds:
                    sel[li][gi, ti] = ov
                for li in touched:
                    conns[li] = sel_to_conn(cand[li], sel[li])
                S[:, start:] = old_suffix
    log(final=True)


if __name__ == "__main__":
    main()
