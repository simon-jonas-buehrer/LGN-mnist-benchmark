"""Optimizer benchmark on a FIXED LUT network: backprop vs coordinate descent vs random
search vs bandit/policy gradient.

The network is identical for every method:

    image (B, 3, 32, 32), light augmentation (random flip + crop-4)
      -> Thermometer encoder           (B, C*b, H, W) threshold bits, flattened to (B, I)
      -> depth x width LUT layers      every node fan-in 4, a 16-entry truth table
      -> GroupSum head                 (B, h*C) bits -> C class logits (popcount / sqrt(h))
      -> cross-entropy                 the one loss/metric (perplexity = exp(loss))

Wiring is the SAME deterministic MONARCH pattern for all methods: nodes form a (G, N/G)
grid; even layers read within their group (block-diagonal), odd layers read across groups
at their own intra-position (the transpose factor), so two layers mix everything. Each tap
additionally has k CANDIDATE sources -- candidate 0 is the monarch tap, candidates 1..k-1
are fixed random alternates shared by all methods (same topology seed). With
--learn-conn 0 every method is frozen to candidate 0; with --learn-conn 1 connections are
learned method-natively:

    bp   straight-through softmax over the k candidates per tap
    cd   present the k candidates for one random tap per visited node, evaluate each
         independently (one exact batch loss per candidate), keep the best if it improves
    rs   candidate re-draws mutate jointly WITH truth-table bit flips (one genome mutation)
    mab  a categorical policy over the k candidates per tap, REINFORCE update

Truth-table initialization (--res-init 1, default): RESIDUAL initialization after
arXiv:2510.03250 -- every gate starts as a PASS-THROUGH of its tap 0 (the monarch
residual connection), so signal flows through depth from step 0 and sign-symmetry is
broken. Method-natively: cd and rs get the exact pass-through table T[cell] = cell & 1;
bp is only BIASED toward it (deterministic latents +-RES_BP around the sin bit, plus tiny
noise), since the relaxation must keep gradients; mab's Bernoulli policy logits are set
to +-RES_MAB (sigmoid ~ 0.88 toward the pass-through bit -- the policy analog of the
paper's bias, softened so REINFORCE can escape). --res-init 0 falls back to Gaussian
latents (bp, mab) /
Bernoulli(0.5) tables (cd, rs). The x-axis of every learning curve is SAMPLES SEEN: each
forward of a training batch of B images costs B (partial recomputes from layer l cost
B * (L - l) / L). Every method logs train/val loss, accuracy and perplexity to
<out>.jsonl and checkpoints to <out>.pt at every eval (and finally with test metrics).

    .venv/bin/python scratch/opt.py --method bp  --learn-conn 1 --out scratch/runs/bp_conn1
    .venv/bin/python scratch/opt.py --method cd  --learn-conn 0 --out scratch/runs/cd_conn0
    sbatch .local/optbench.sbatch      # the full 4 methods x {fixed, learnable} array
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Thermometer, ste_bit  # noqa: E402
from train import load_cifar10  # noqa: E402

CLS = 10
TOPO_SEED = 0  # wiring + candidate sources: identical for every method, always
RES_BP = 1.2   # bp residual bias on the sin latent: soft bit ~ 0.97 toward pass-through
RES_MAB = 1.5  # mab residual bias on the Bernoulli policy logit (sigmoid(1.5) ~ 0.82).
               # The paper's logit-5 is for BACKPROP gate logits; for a sampling policy
               # the per-bit REINFORCE escape gradient is lr*a*(b-p) ~ 0.0007/step at
               # p=0.993 -- measured at full scale: loss 24.4->24.0 in 7k steps, frozen.


def res_pattern(fan_in: int) -> torch.Tensor:
    """(2**fan_in,) bool: the truth table of 'pass through tap 0' (cell bit 0)."""
    return (torch.arange(1 << fan_in) & 1).bool()


# ==========================================================================================
# Topology: monarch wiring + k fixed candidate sources per tap
# ==========================================================================================
def monarch_groups(in_dim: int, out_dim: int, fan_in: int) -> int:
    """Largest power-of-two group count G <= 256 dividing both dims (block size >= fan_in)."""
    g = 256
    while g > 1 and (in_dim % g or out_dim % g or in_dim // g < fan_in):
        g //= 2
    assert g >= fan_in, f"no monarch grouping for {in_dim}->{out_dim}"
    return g


def build_candidates(in_dims: list[int], widths: list[int], fan_in: int, k: int,
                     device: str) -> list[torch.Tensor]:
    """Per layer: (N, fan_in, k) int64 candidate sources. Candidate 0 is the monarch tap:
    even layers block-diagonal (node (r, c) reads group r), odd layers across groups (node
    (r, c) reads position c of groups r + t*G/fan_in). Candidates 1..k-1 are uniform random
    over the layer's input, drawn from a fixed CPU generator so every method (and every
    seed) sees the exact same candidate sets."""
    gen = torch.Generator().manual_seed(TOPO_SEED)
    cands = []
    for l, (i_dim, n) in enumerate(zip(in_dims, widths)):
        g = monarch_groups(i_dim, n, fan_in)
        ipg, opg = i_dim // g, n // g
        j = torch.arange(n)
        r, c = j // opg, j % opg
        t = torch.arange(fan_in)
        if l % 2 == 0:  # within-group: block-diagonal factor
            mon = r[:, None] * ipg + (c[:, None] * fan_in + t[None]) % ipg
        else:           # across-group: the transpose factor
            mon = ((r[:, None] + t[None] * (g // fan_in)) % g) * ipg + (c % ipg)[:, None]
        cand = torch.randint(i_dim, (n, fan_in, k), generator=gen)
        cand[:, :, 0] = mon
        cands.append(cand.to(device))
    return cands


# ==========================================================================================
# Hard executor (cd / rs / mab): uint8 bits all the way through
# ==========================================================================================
def sel_to_conn(cand: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
    """(N, F, k) candidates + (N, F) selection -> (N, F) concrete sources."""
    return cand.gather(2, sel[:, :, None]).squeeze(2)


def _tt_apply(tt: torch.Tensor, cell: torch.Tensor, rows: int = 512) -> torch.Tensor:
    """tt (N, 16) uint8, cell (B, N) uint8 -> out[b, n] = tt[n, cell[b, n]], row-chunked
    so the int64 index temporary stays small."""
    flat = tt.reshape(-1)
    base = torch.arange(tt.shape[0], device=tt.device, dtype=torch.int64) * tt.shape[1]
    out = torch.empty_like(cell)
    for i in range(0, cell.shape[0], rows):
        out[i:i + rows] = flat[base + cell[i:i + rows].long()]
    return out


@torch.no_grad()
def fwd_hard(x0: torch.Tensor, conns: list[torch.Tensor], tts: list[torch.Tensor],
             from_layer: int = 0, acts: list[torch.Tensor] | None = None) -> list[torch.Tensor]:
    """x0 (B, I) uint8 bits. Returns acts, length L+1: acts[l] is the input to layer l,
    acts[-1] the final bits. With from_layer > 0 the prefix of `acts` is reused by
    reference and only the suffix is recomputed (fresh tensors -- rejecting a proposal is
    just dropping the returned list)."""
    new = [x0] if acts is None else list(acts[:from_layer + 1])
    x = new[-1]
    for l in range(from_layer, len(conns)):
        conn = conns[l]
        cell = x[:, conn[:, 0]].clone()
        for t in range(1, conn.shape[1]):
            cell |= x[:, conn[:, t]] << t
        x = _tt_apply(tts[l], cell)
        new.append(x)
    return new


def head_loss(bits: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, h*CLS) bits -> (CE loss, logits): group popcount / sqrt(h) (GroupSum head).
    Summed in int32 in batch chunks so the (B, width) upcast never materializes at once
    (a full-width float32 copy is ~2 GB at batch 8192, width 64000)."""
    B, h = bits.shape[0], bits.shape[1] // CLS
    logits = torch.empty(B, CLS, device=bits.device)
    for i in range(0, B, 2048):
        logits[i:i + 2048] = bits[i:i + 2048].view(-1, CLS, h).sum(-1, dtype=torch.int32)
    logits /= math.sqrt(h)
    return F.cross_entropy(logits, y), logits


# ==========================================================================================
# Differentiable model (bp): hard bits forward, sin/softmax surrogates backward
# ==========================================================================================
def st_onehot(alpha: torch.Tensor) -> torch.Tensor:
    """Straight-through categorical: one-hot argmax forward, softmax gradient backward."""
    p = alpha.softmax(-1)
    hard = F.one_hot(alpha.argmax(-1), alpha.shape[-1]).to(p.dtype)
    return hard + p - p.detach()


class BPNet(nn.Module):
    """The same LUT stack with real latents: truth tables theta ~ N(0,1) via the sin
    straight-through bit, connections a straight-through softmax over the k candidates.
    Because both surrogates are HARD in the forward pass, this module computes exactly the
    same boolean-circuit family as the hard executor -- only the gradients are soft."""

    def __init__(self, cands: list[torch.Tensor], fan_in: int, k: int, learn_conn: bool,
                 gen: torch.Generator, res_init: bool):
        super().__init__()
        self.L, self.fan_in, self.k, self.learn_conn = len(cands), fan_in, k, learn_conn
        self.theta = nn.ParameterList()
        self.alpha = nn.ParameterList()
        for l, cand in enumerate(cands):
            self.register_buffer(f"cand{l}", cand)
            n = cand.shape[0]
            if res_init:  # biased toward the tap-0 pass-through table (residual init)
                t = RES_BP * (res_pattern(fan_in).float() * 2 - 1).expand(n, -1) \
                    + 0.05 * torch.randn(n, 1 << fan_in, generator=gen)
            else:
                t = torch.randn(n, 1 << fan_in, generator=gen)
            self.theta.append(nn.Parameter(t))
            if learn_conn:
                a = torch.randn(n, fan_in, k, generator=gen)
                a[:, :, 0] += 4.0  # start at the monarch tap, like everyone else
                self.alpha.append(nn.Parameter(a))

    def layer(self, x: torch.Tensor, l: int) -> torch.Tensor:
        cand = getattr(self, f"cand{l}")
        if self.learn_conn:
            w = st_onehot(self.alpha[l])                       # (N, F, k)
            a = x[:, cand[:, :, 0]] * w[:, :, 0]
            for j in range(1, self.k):
                a = a + x[:, cand[:, :, j]] * w[:, :, j]       # (B, N, F)
        else:
            a = x[:, cand[:, :, 0]]
        y = ste_bit(self.theta[l]).unsqueeze(0)                # (1, N, 16) hard bits
        for i in reversed(range(self.fan_in)):                 # multilinear contraction
            half = 1 << i
            ai = a[:, :, i].unsqueeze(-1)                      # (B, N, 1)
            y = y[..., :half] * (1 - ai) + y[..., half:] * ai
        return y.squeeze(-1)                                   # (B, N)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l in range(self.L):
            if self.training:
                x = checkpoint(self.layer, x, l, use_reentrant=False)
            else:
                x = self.layer(x, l)
        h = x.shape[1] // CLS
        return x.view(x.shape[0], CLS, h).sum(-1) / math.sqrt(h)


# ==========================================================================================
# Data: augmentation + encoding
# ==========================================================================================
def augment(x: torch.Tensor, crop: int = 4) -> torch.Tensor:
    """Light augmentation on (B, 3, 32, 32) in [0,1]: random horizontal flip + random
    crop from `crop`-pixel replicate padding."""
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
# Shared harness: samples-seen accounting, eval, jsonl log, checkpoints
# ==========================================================================================
class Bench:
    def __init__(self, args):
        self.args = args
        self.dev = args.device
        torch.manual_seed(args.seed)
        tx, ty, ex, ey = load_cifar10(args.data_dir, args.download)
        nv = 5000
        self.px, self.py = tx[:-nv].to(self.dev), ty[:-nv].to(self.dev)
        self.enc = Thermometer(num_bits=args.num_bits).fit(tx[:2000]).to(self.dev)
        self.I = 3 * args.num_bits * 32 * 32
        self.widths = [args.width] * args.depth
        in_dims = [self.I] + self.widths[:-1]
        self.cands = build_candidates(in_dims, self.widths, args.fan_in, args.k, self.dev)
        # fixed eval views (never augmented)
        self.evals = {"train": (self.encode(self.px[:5000]), self.py[:5000]),
                      "val": (self.encode(tx[-nv:].to(self.dev)), ty[-nv:].to(self.dev))}
        self.test = (ex, ey)  # encoded lazily at the final eval only
        self.samples = 0.0
        self.t0 = time.time()
        self.wall_offset = 0.0          # cumulative minutes from earlier sessions (resume)
        self.last_eval = None
        self.out = args.out
        self.out.parent.mkdir(parents=True, exist_ok=True)
        # Resume by default: if a checkpoint already exists at this --out, reload it (weights
        # loaded per-method below), continue the samples / walltime axes, and APPEND to the
        # jsonl so pause-and-continue is seamless. No flag -- to start over, delete the .pt.
        # The stop budget (--max-minutes) is fresh each session; only the logged axes accumulate.
        self.resume_ckpt = None
        pt = self.out.with_suffix(".pt")
        if pt.exists():
            self.resume_ckpt = torch.load(pt, map_location=self.dev, weights_only=False)
            self.samples = float(self.resume_ckpt.get("samples", 0))
            jl = self.out.with_suffix(".jsonl")
            lines = [ln for ln in jl.read_text().splitlines() if ln.strip()] if jl.exists() else []
            self.wall_offset = json.loads(lines[-1])["min"] if lines else 0.0
            print(f"RESUMED {pt.name}: samples={int(self.samples):,} "
                  f"wall_offset={self.wall_offset:.1f}m", flush=True)
        else:
            self.out.with_suffix(".jsonl").write_text("")
        print(f"method={args.method} learn_conn={args.learn_conn} I={self.I} "
              f"width={args.width} depth={args.depth} fan_in={args.fan_in} k={args.k} "
              f"batch={args.batch} train={len(self.px)} val=5000", flush=True)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.enc(images).flatten(1).to(torch.uint8)

    def batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(len(self.px), (self.args.batch,), device=self.dev)
        return self.encode(augment(self.px[idx], self.args.crop)), self.py[idx]

    @staticmethod
    def metrics(loss_sum: float, correct: int, n: int) -> dict:
        loss = loss_sum / n
        return {"loss": round(loss, 4), "acc": round(100.0 * correct / n, 2),
                "ppl": round(math.exp(min(loss, 30.0)), 4)}

    def eval_sets(self, loss_fn, final: bool = False) -> dict:
        """loss_fn(xb_u8, yb) -> (sum CE, n correct); chunked over each eval split."""
        sets = dict(self.evals)
        if final:
            ex, ey = self.test
            sets["test"] = (self.encode(ex.to(self.dev)), ey.to(self.dev))
        out = {}
        for name, (xe, ye) in sets.items():
            ls, co = 0.0, 0
            for i in range(0, len(xe), 2048):
                l, c = loss_fn(xe[i:i + 2048], ye[i:i + 2048])
                ls, co = ls + l, co + c
            out[name] = self.metrics(ls, co, len(xe))
        return out

    def log(self, loss_fn, save_fn, extra: dict, final: bool = False) -> None:
        m = self.eval_sets(loss_fn, final=final)
        rec = {"samples": int(self.samples),
               "min": round(self.wall_offset + (time.time() - self.t0) / 60, 1),
               **{k: v for k, v in m.items()}, **extra}
        with open(self.out.with_suffix(".jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        save_fn(self.out.with_suffix(".pt"))
        te = f" test {m['test']['loss']:.3f}/{m['test']['acc']:5.2f}" if final else ""
        print(f"{int(self.samples):>12,} | train {m['train']['loss']:.3f}/"
              f"{m['train']['acc']:5.2f} | val {m['val']['loss']:.3f}/"
              f"{m['val']['acc']:5.2f} | ppl {m['val']['ppl']:6.2f}{te} | "
              f"{rec['min']:6.1f}m {extra}", flush=True)

    def due(self) -> bool:
        now = time.time()
        if self.last_eval is None or now - self.last_eval >= self.args.eval_mins * 60:
            self.last_eval = now
            return True
        return False

    def done(self) -> bool:
        if self.samples >= self.args.max_samples:
            return True
        return bool(self.args.max_minutes) and \
            (time.time() - self.t0) / 60 >= self.args.max_minutes


# ==========================================================================================
# Trainer 1: naive backprop (Adam on the straight-through relaxation)
# ==========================================================================================
def train_bp(b: Bench) -> None:
    args = b.args
    gen = torch.Generator().manual_seed(args.seed)
    model = BPNet(b.cands, args.fan_in, args.k, bool(args.learn_conn), gen,
                  bool(args.res_init)).to(b.dev)
    if b.resume_ckpt is not None:
        model.load_state_dict({k: v.to(b.dev) for k, v in b.resume_ckpt["state"].items()})
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"bp params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    @torch.no_grad()
    def loss_fn(xe, ye):
        model.eval()
        ls, co = 0.0, 0  # sub-chunk: the soft eval materializes (B, N, ...) floats
        for i in range(0, len(xe), 256):
            logits = model(xe[i:i + 256].float())
            ls += F.cross_entropy(logits, ye[i:i + 256], reduction="sum").item()
            co += (logits.argmax(1) == ye[i:i + 256]).sum().item()
        return ls, co

    def save_fn(p):
        torch.save({"method": "bp", "args": vars(args) | {"out": str(args.out)},
                    "samples": int(b.samples),
                    "state": {k: v.cpu() for k, v in model.state_dict().items()}}, p)

    step, lr = 0, args.lr
    while not b.done():
        if b.due():
            b.log(loss_fn, save_fn, {"step": step, "lr": round(lr, 5)})
        if args.max_minutes:  # cosine decay over the walltime budget (plateaued flat
            frac = min(1.0, (time.time() - b.t0) / 60 / args.max_minutes)  # at lr 1e-2)
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * frac))
            for g in opt.param_groups:
                g["lr"] = lr
        model.train()
        xb, yb = b.batch()
        loss = F.cross_entropy(model(xb.float()), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        b.samples += args.batch
        step += 1
    b.log(loss_fn, save_fn, {"step": step, "lr": round(lr, 5)}, final=True)


# ==========================================================================================
# Hard-state helpers (cd / rs / mab share the executor)
# ==========================================================================================
def bern_state(b: Bench, gen: torch.Generator) -> tuple[list, list, list]:
    """Truth tables (exact tap-0 pass-through with --res-init, else Bernoulli(0.5)) +
    monarch selection (candidate 0 everywhere)."""
    if b.args.res_init:
        pat = res_pattern(b.args.fan_in).to(torch.uint8).to(b.dev)
        tts = [pat.expand(w, -1).clone() for w in b.widths]
    else:
        tts = [(torch.rand(w, 16, generator=gen) < 0.5).to(torch.uint8).to(b.dev)
               for w in b.widths]
    sels = [torch.zeros(w, b.args.fan_in, dtype=torch.int64, device=b.dev)
            for w in b.widths]
    conns = [sel_to_conn(c, s) for c, s in zip(b.cands, sels)]
    return tts, sels, conns


def load_hard_state(b: Bench, tts, sels):
    """If resuming, overwrite the freshly-initialized hard genome with the checkpointed
    truth tables + tap selections. Returns (tts, sels, conns)."""
    if b.resume_ckpt is not None:
        tts = [t.to(b.dev) for t in b.resume_ckpt["tt"]]
        sels = [s.to(b.dev) for s in b.resume_ckpt["sel"]]
    conns = [sel_to_conn(c, s) for c, s in zip(b.cands, sels)]
    return tts, sels, conns


def hard_loss_fn(conns, tts):
    @torch.no_grad()
    def loss_fn(xe, ye):
        loss, logits = head_loss(fwd_hard(xe, conns, tts)[-1], ye)
        return loss.item() * len(ye), (logits.argmax(1) == ye).sum().item()
    return loss_fn


def hard_save_fn(b: Bench, name: str, tts, sels, extra=None):
    def save_fn(p):
        torch.save({"method": name, "args": vars(b.args) | {"out": str(b.args.out)},
                    "samples": int(b.samples),
                    "tt": [t.cpu() for t in tts], "sel": [s.cpu() for s in sels]}
                   | ({k: [v.cpu() for v in vs] for k, vs in extra.items()} if extra else {}),
                   p)
    return save_fn


# ==========================================================================================
# Trainer 2: pure coordinate descent (exact batch-loss accepts, block coordinates)
# ==========================================================================================
def train_cd(b: Bench) -> None:
    """Plain block coordinate descent. Per proposal: pick a random layer, flip one random
    truth-table cell on a random chunk of --chunk nodes, recompute, and KEEP the flip iff
    the loss on the (large) batch strictly decreases, else revert. With --learn-conn one
    proposal in four instead re-taps a chunk of nodes to a random candidate and keeps it
    iff the loss decreases. That is the whole method -- no gradients, no acceptance test
    beyond 'did the number go down'. The batch is held fixed for --props proposals so
    accepted flips compound on one objective, then re-rolled; --batch is large enough
    that a batch improvement is a real improvement (the only knob that matters). Partial
    recomputes from layer l cost B*(L-l)/L samples."""
    args = b.args
    gen = torch.Generator().manual_seed(args.seed)
    tts, sels, conns = bern_state(b, gen)
    tts, sels, conns = load_hard_state(b, tts, sels)
    L = args.depth
    accepts = trials = 0

    def loss_from(l, acts, yb):
        b.samples += args.batch * (L - l) / L
        new = fwd_hard(acts[0], conns, tts, from_layer=l, acts=acts)
        return new, head_loss(new[-1], yb)[0].item()

    while not b.done():
        if b.due():
            b.log(hard_loss_fn(conns, tts), hard_save_fn(b, "cd", tts, sels),
                  {"accepts": accepts, "trials": trials})
        xb, yb = b.batch()
        acts = fwd_hard(xb, conns, tts)
        base = head_loss(acts[-1], yb)[0].item()
        b.samples += args.batch
        for p in range(args.props):
            trials += 1
            l = int(torch.randint(L, (1,)))
            idx = torch.randperm(b.widths[l], device=b.dev)[:args.chunk]
            if args.learn_conn and p % 4 == 0:               # connection move
                tap = torch.randint(args.fan_in, (idx.numel(),), device=b.dev)
                old = sels[l][idx, tap].clone()
                sels[l][idx, tap] = torch.randint(args.k, (idx.numel(),), device=b.dev)
                conns[l] = sel_to_conn(b.cands[l], sels[l])
                new, lp = loss_from(l, acts, yb)
                if lp < base:
                    acts, base, accepts = new, lp, accepts + 1
                else:
                    sels[l][idx, tap] = old
                    conns[l] = sel_to_conn(b.cands[l], sels[l])
            else:                                            # truth-table flip
                cell = torch.randint(16, (idx.numel(),), device=b.dev)
                tts[l] = tts[l].clone()
                tts[l][idx, cell] ^= 1
                new, lp = loss_from(l, acts, yb)
                if lp < base:
                    acts, base, accepts = new, lp, accepts + 1
                else:
                    tts[l][idx, cell] ^= 1
    b.log(hard_loss_fn(conns, tts), hard_save_fn(b, "cd", tts, sels),
          {"accepts": accepts, "trials": trials}, final=True)


# ==========================================================================================
# Trainer 3: pure random search ((1+1)-ES on the discrete genome)
# ==========================================================================================
def train_rs(b: Bench) -> None:
    """Plain (1+1) random search / hill climbing. Per step: copy the genome, flip --rs-tt
    random truth-table cells (and with --learn-conn re-tap --rs-conn random taps), score
    current and mutant on the SAME batch, and keep the mutant iff its loss is strictly
    lower. No step-size adaptation, no statistics -- just 'mutate, keep if better'. The
    only knob is --rs-tt (how many cells to flip) and --batch (large so a batch win is a
    real win). Costs 2B samples per step."""
    args = b.args
    gen = torch.Generator().manual_seed(args.seed)
    tts, sels, conns = bern_state(b, gen)
    tts, sels, conns = load_hard_state(b, tts, sels)
    L = args.depth
    accepts = trials = 0

    def loss(cs, ts, xb, yb):
        return head_loss(fwd_hard(xb, cs, ts)[-1], yb)[0].item()

    while not b.done():
        if b.due():
            b.log(hard_loss_fn(conns, tts), hard_save_fn(b, "rs", tts, sels),
                  {"accepts": accepts, "trials": trials})
        xb, yb = b.batch()
        cur = loss(conns, tts, xb, yb)
        m_tts, m_sels, m_conns = list(tts), list(sels), list(conns)
        l = int(torch.randint(L, (1,)))                      # one random layer per mutation
        m_tts[l] = m_tts[l].clone()                          # flip rs_tt random cells in it
        m_tts[l][torch.randint(b.widths[l], (args.rs_tt,), device=b.dev),
                 torch.randint(16, (args.rs_tt,), device=b.dev)] ^= 1
        if args.learn_conn:
            m_sels[l] = m_sels[l].clone()                    # re-tap rs_conn random taps
            m_sels[l][torch.randint(b.widths[l], (args.rs_conn,), device=b.dev),
                      torch.randint(args.fan_in, (args.rs_conn,), device=b.dev)] = \
                torch.randint(args.k, (args.rs_conn,), device=b.dev)
            m_conns[l] = sel_to_conn(b.cands[l], m_sels[l])
        mut = loss(m_conns, m_tts, xb, yb)
        b.samples += 2 * args.batch
        trials += 1
        if mut < cur:
            tts, sels, conns = m_tts, m_sels, m_conns
            accepts += 1
    b.log(hard_loss_fn(conns, tts), hard_save_fn(b, "rs", tts, sels),
          {"accepts": accepts, "trials": trials}, final=True)


# ==========================================================================================
# Trainer 4: pure multi-armed bandit / policy gradient (REINFORCE per bit and per tap)
# ==========================================================================================
def train_mab(b: Bench) -> None:
    """Per-node policy gradient (REINFORCE) with a LEAVE-ONE-OUT baseline (RLOO) --
    nothing like backprop. Every node is a stochastic policy over the function it
    implements: each of its 16 truth-table cells is an independent Bernoulli arm with a
    logit (theta), and -- with --learn-conn -- each tap is a k-armed categorical arm
    (logits alpha). Per step: sample --mab-rollouts INDEPENDENT whole circuits from the
    policy, run each (reward R_i = -CE), and give rollout i the advantage
    A_i = R_i - mean_{j!=i} R_j (leave-one-out: the other rollouts ARE its baseline).
    Then the score-function update summed over rollouts and every node at once:

        theta += lr/N * sum_i A_i * (bit_i - p)             # Bernoulli cell
        alpha += lr/N * sum_i A_i * (onehot(a_i) - pi)      # categorical tap

    RLOO is the key variance-reduction fix (arXiv:2402.14740 and the classic VIMCO): a
    single global scalar reward shared across ~8M cells gives a hopelessly noisy per-node
    gradient (measured: pure single-rollout REINFORCE stalled ~13-15% val); the LOO
    baseline is the minimum-variance unbiased control variate for this exchangeable-
    sample estimator and cuts the noise ~N-fold at N-forward cost. Advantages are
    additionally std-normalized across the rollout batch. Greedy circuit (sigmoid>0.5 /
    argmax) is evaluated and checkpointed. Costs N*B samples per step."""
    args = b.args
    gen = torch.Generator().manual_seed(args.seed)
    thetas = [(0.5 * torch.randn(w, 16, generator=gen)).to(b.dev) for w in b.widths]
    alphas = [(0.5 * torch.randn(w, args.fan_in, args.k, generator=gen)).to(b.dev)
              for w in b.widths]
    for a in alphas:
        a[:, :, 0] += 3.0  # bandit prior: start near the monarch tap
    if b.resume_ckpt is not None:
        thetas = [t.to(b.dev) for t in b.resume_ckpt["theta"]]
        alphas = [a.to(b.dev) for a in b.resume_ckpt["alpha"]]
    N, step = args.mab_rollouts, 0

    def greedy():
        tts = [(t > 0).to(torch.uint8) for t in thetas]
        sels = [a.argmax(-1) for a in alphas] if args.learn_conn else \
            [torch.zeros(w, args.fan_in, dtype=torch.int64, device=b.dev) for w in b.widths]
        return tts, sels, [sel_to_conn(c, s) for c, s in zip(b.cands, sels)]

    def save_fn(p):
        tts, sels, _ = greedy()
        hard_save_fn(b, "mab", tts, sels, extra={"theta": thetas, "alpha": alphas})(p)

    while not b.done():
        if b.due():
            tts, sels, conns = greedy()
            b.log(hard_loss_fn(conns, tts), save_fn, {"step": step})
        xb, yb = b.batch()
        probs = [torch.sigmoid(t) for t in thetas]
        pis = [a.softmax(-1) for a in alphas] if args.learn_conn else None
        rolls, R = [], []                                    # N independent rollouts
        for _ in range(N):
            bits = [(torch.rand_like(p) < p) for p in probs]
            tts = [bt.to(torch.uint8) for bt in bits]
            if args.learn_conn:
                acts = [torch.multinomial(pi.reshape(-1, args.k), 1).reshape(w, args.fan_in)
                        for pi, w in zip(pis, b.widths)]
            else:
                acts = [torch.zeros(w, args.fan_in, dtype=torch.int64, device=b.dev)
                        for w in b.widths]
            conns = [sel_to_conn(c, s) for c, s in zip(b.cands, acts)]
            R.append(-head_loss(fwd_hard(xb, conns, tts)[-1], yb)[0].item())
            rolls.append((bits, acts))
            b.samples += args.batch
        Rt = torch.tensor(R)
        adv = Rt - (Rt.sum() - Rt) / (N - 1)                 # leave-one-out baseline
        adv = (adv / (adv.std() + 1e-8)).tolist()            # normalize across rollouts
        for i, (bits, acts) in enumerate(rolls):             # REINFORCE, LOO-baselined
            for l in range(args.depth):
                thetas[l] += args.lr_mab / N * adv[i] * (bits[l].float() - probs[l])
                if args.learn_conn:
                    alphas[l] += args.lr_mab / N * adv[i] \
                        * (F.one_hot(acts[l], args.k).float() - pis[l])
        step += 1
    tts, sels, conns = greedy()
    b.log(hard_loss_fn(conns, tts), save_fn, {"step": step}, final=True)


# ==========================================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["bp", "cd", "rs", "mab"], required=True)
    p.add_argument("--learn-conn", type=int, default=1,
                   help="1: learn connections over the k candidates (method-natively); "
                        "0: frozen monarch wiring (candidate 0)")
    p.add_argument("--data-dir", type=Path, default=Path("data/cifar-10-batches-py"))
    p.add_argument("--download", action="store_true")
    p.add_argument("--num-bits", type=int, default=4, help="thermometer bits per channel")
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--width", type=int, default=64000,
                   help="nodes per layer ('64K'; divisible by 10 classes and 256 groups)")
    p.add_argument("--fan-in", type=int, default=4)
    p.add_argument("--k", type=int, default=8, help="candidate sources per tap")
    p.add_argument("--res-init", type=int, default=1,
                   help="1: residual init (arXiv:2510.03250) -- every gate starts as a "
                        "pass-through of tap 0 (exact for cd/rs, biased latents for bp, "
                        "biased policy for mab); 0: random init")
    p.add_argument("--crop", type=int, default=4, help="augmentation crop padding")
    p.add_argument("--batch", type=int, default=0, help="0 = per-method default")
    p.add_argument("--max-samples", type=float, default=100e6)
    p.add_argument("--max-minutes", type=float, default=0,
                   help="stop cleanly (final eval + test + ckpt) after this walltime; 0=off")
    p.add_argument("--eval-mins", type=float, default=2.0,
                   help="wall-clock minutes between evals (same curve density for slow "
                        "and fast methods)")
    p.add_argument("--lr", type=float, default=1e-2, help="bp Adam lr")
    p.add_argument("--lr-mab", type=float, default=1.5, help="mab policy lr")
    p.add_argument("--chunk", type=int, default=1024, help="cd nodes per proposal")
    p.add_argument("--props", type=int, default=128,
                   help="cd proposals per batch re-roll: the batch objective is held "
                        "fixed this long so accepted gains compound")
    p.add_argument("--halvings", type=int, default=6, help="cd package halvings on reject")
    p.add_argument("--rs-tt", type=int, default=16,
                   help="rs table cells flipped per mutation (fewer = higher accept rate)")
    p.add_argument("--rs-conn", type=int, default=16,
                   help="rs base tap re-draws per mutation (scaled by f)")
    p.add_argument("--rs-z", type=float, default=1.5,
                   help="rs acceptance confidence: accept if mean(d) < -z*std/sqrt(B)")
    p.add_argument("--rs-lambda", type=int, default=8,
                   help="rs (1+lambda): mutants scored per base eval, sharing its forward")
    p.add_argument("--mab-rollouts", type=int, default=8,
                   help="mab RLOO rollouts per step; each is baselined by the others' mean "
                        "(variance reduction ~N-fold at N-forward cost)")
    p.add_argument("--out", type=Path, required=True, help="prefix for .jsonl and .pt; an "
                   "existing .pt is resumed automatically (delete it to start fresh)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    assert args.width % CLS == 0, "width must be divisible by the class count"
    if args.batch == 0:
        args.batch = {"bp": 256, "cd": 4096, "rs": 8192, "mab": 8192}[args.method]
    print(f"args={vars(args)}", flush=True)
    bench = Bench(args)
    {"bp": train_bp, "cd": train_cd, "rs": train_rs, "mab": train_mab}[args.method](bench)


if __name__ == "__main__":
    main()
