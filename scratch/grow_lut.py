"""Backprop-free LUT network grown by greedy correlation + coordinate descent.

A scratch experiment: build the same kind of boolean-gate circuit as model.py, but with no
gradients at all. Everything is binary and bitpacked, so building and inference are a few GPU
matmuls and bit ops.

Architecture: one window
------------------------
There is a single fixed window of WIN = f * N signal slots (f defaults to 4):

    slot 0 .. N-1        the N input bits from the thermometer encoder (frozen)
    slot N .. WIN-1      start at 0, filled by gates as we build

Every slot is also a head output bit. The GroupSum head reads the *whole window*: a slot's
class is slot % C (round-robin, so the input bits spread evenly over the 10 classes), each
class owns H = WIN / C slots, and the class score is the popcount of its slots, divided by
tau = sqrt(H). Because H * C = WIN = f * N, the "(B,N) -> (B,H*C)" map is just "fill the
window". This unifies the signal pool, the output bits and the depth wiring into one array.

Building, no backprop
---------------------
Each build sweep, on a random batch:

  1. residual per slot in {-1,0,+1}: a class-c slot wants to fire more where class c should go
     up (multiclass-hinge subgradient) but the slot is currently 0, and fire less where class c
     should go down but the slot is currently 1. So every slot chases its own mistakes.
  2. correlate every filled slot against every buildable slot's residual: a
     (filled x buildable) ~ f*N^2 covariance matrix, one matmul (chunked over targets).
  3. for the strongest target slots, take the top-2 correlating signals and wire an AND/OR gate
     with NOTs on the negatively correlated inputs.

Empty slots have the biggest residual, so they fill first; later sweeps refine filled slots.
New gates can read any filled slot, including earlier gates, which gives depth.

Coordinate descent
-------------------
Greedy wiring overfits the batch it saw, so we periodically run CD: take a batch and, for a
sample of gate slots, try flipping each of the 4 truth-table bits; keep the flip that most
lowers the batch hinge loss. Because the head sums the window, flipping one slot's gate only
changes that slot's class score, so the loss delta is exact and computed in closed form for all
candidate gates and all 4 flips at once. CD explores all 16 two-input functions, not just the
AND/OR we started from.

Over the run we ramp from "lots of build, little CD" to "little build, lots of CD".

Run
---
    .venv/bin/python scratch/grow_lut.py --device cuda --train-size 0
    bash scratch/run_srun.sh            # interactive single GPU
    sbatch scratch/run.sbatch           # batch single GPU
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Thermometer  # noqa: E402
from train import load_cifar10  # noqa: E402

# The 16 two-input boolean functions, indexed by f00*8 + f01*4 + f10*2 + f11 (the truth table
# [f00,f01,f10,f11] read as a 4-bit number). Used to label what each grown gate became.
FN_NAMES = ["FALSE", "AND", "a&!b", "a", "!a&b", "b", "XOR", "OR",
            "NOR", "XNOR", "!b", "a|!b", "!a", "!a|b", "NAND", "TRUE"]


# ======================================================================================
# Bitpacking: a signal is a binary value over all D samples, stored as ceil(D/64) int64 words
# ======================================================================================
_SHIFTS = None


def _shifts(device: torch.device) -> torch.Tensor:
    global _SHIFTS
    if _SHIFTS is None or _SHIFTS.device != device:
        _SHIFTS = torch.arange(64, dtype=torch.int64, device=device)
    return _SHIFTS


def pack_bits(bits: torch.Tensor) -> torch.Tensor:
    """(n, D) of {0,1} -> (n, W) int64. Disjoint bit positions, so summing == OR in two's compl."""
    n, d = bits.shape
    w = (d + 63) // 64
    pad = w * 64 - d
    if pad:
        bits = torch.cat([bits, bits.new_zeros(n, pad)], dim=1)
    bits = bits.to(torch.int64).view(n, w, 64)
    return (bits << _shifts(bits.device)).sum(-1)


def unpack_bits(words: torch.Tensor, d: int, word_chunk: int = 32) -> torch.Tensor:
    """(n, W) int64 -> (n, D) uint8. (x >> s) & 1 recovers bit s for s in 0..63. Done in word
    chunks and cast to uint8 immediately so the int64 (n, chunk, 64) temporary stays small
    (unpacking a whole large window at once otherwise blows up to many GB)."""
    n, w = words.shape
    sh = _shifts(words.device)
    out = torch.empty((n, w * 64), dtype=torch.uint8, device=words.device)
    for c0 in range(0, w, word_chunk):
        wc = words[:, c0:c0 + word_chunk]
        bitc = ((wc.unsqueeze(-1) >> sh) & 1).to(torch.uint8)      # (n, cw, 64)
        out[:, c0 * 64:(c0 + wc.shape[1]) * 64] = bitc.reshape(n, -1)
    return out[:, :d]


def gather_batch(words: torch.Tensor, slots: torch.Tensor, bidx: torch.Tensor) -> torch.Tensor:
    """Bits of the given signal slots on the given samples -> (len(slots), len(bidx)) float."""
    word = bidx >> 6
    off = (bidx & 63).to(torch.int64)
    sel = words[slots][:, word]
    return ((sel >> off) & 1).to(torch.float32)


def apply_gate(a: torch.Tensor, b: torch.Tensor, tt: torch.Tensor) -> torch.Tensor:
    """Packed 2-input LUT. a,b: (k, W) int64 signal words; tt: (k, 4) bool truth table
    [f00,f01,f10,f11] indexed by a_bit*2 + b_bit. Returns (k, W) int64."""
    na, nb = ~a, ~b
    f00, f01, f10, f11 = tt[:, 0:1], tt[:, 1:2], tt[:, 2:3], tt[:, 3:4]
    out = torch.where(f00, na & nb, a.new_zeros(()))
    out = out | torch.where(f01, na & b, a.new_zeros(()))
    out = out | torch.where(f10, a & nb, a.new_zeros(()))
    out = out | torch.where(f11, a & b, a.new_zeros(()))
    return out


# ======================================================================================
# The grown circuit
# ======================================================================================
class GrownCircuit:
    def __init__(self, n_inputs: int, n_classes: int, window_factor: float,
                 *, gate: str, max_gates: int, device: str):
        self.N = n_inputs
        self.C = n_classes
        win = int(round(window_factor * n_inputs))
        win -= win % n_classes                       # make it a clean multiple of C
        self.WIN = win
        self.H = win // n_classes
        self.tau = math.sqrt(self.H)
        self.gate = gate
        self.max_gates = max_gates
        self.device = device

        self.class_of = torch.arange(win, device=device) % n_classes   # round-robin grouping
        self.filled = torch.zeros(win, dtype=torch.bool, device=device)
        self.depth = torch.zeros(win, dtype=torch.long, device=device)  # gates stacked below a slot
        self.usage = torch.zeros(win, dtype=torch.long, device=device)  # times used as a gate input
        self.def_a = torch.full((win,), -1, dtype=torch.long, device=device)
        self.def_b = torch.full((win,), -1, dtype=torch.long, device=device)
        self.def_tt = torch.zeros((win, 4), dtype=torch.bool, device=device)
        self.buildable = torch.arange(n_inputs, win, device=device)    # slots gates may write

        self.ops: list[tuple] = []                   # ordered op-batches (slots, a, b, tt)
        self.n_gates_built = 0
        self.win = None                              # (WIN, Wwords) int64 over the train set
        self.score = None                            # (C, D) class scores over the train set

    # -- setup --------------------------------------------------------------------------
    def set_inputs(self, input_bits: torch.Tensor) -> None:
        """input_bits: (N, D) uint8. Loads them into the bottom N window slots."""
        d = input_bits.shape[1]
        self.D = d
        self.Wwords = (d + 63) // 64
        self.win = torch.zeros((self.WIN, self.Wwords), dtype=torch.int64, device=self.device)
        self.win[: self.N] = pack_bits(input_bits.to(self.device))
        self.filled[: self.N] = True
        # head reads the whole window, so the input slots already contribute to the scores
        vals = unpack_bits(self.win[: self.N], d).to(torch.float32)
        self.score = torch.zeros((self.C, d), dtype=torch.float32, device=self.device)
        self.score.index_add_(0, self.class_of[: self.N], vals)

    # -- supervision --------------------------------------------------------------------
    def class_direction(self, y: torch.Tensor, bidx: torch.Tensor):
        """Per-class desired direction in {-1,0,+1} (multiclass-hinge subgradient) plus the
        cached batch logits, returned for reuse."""
        logit = self.score[:, bidx] / self.tau          # (C, Bb)
        bb = bidx.shape[0]
        ar = torch.arange(bb, device=self.device)
        yb = y[bidx]                                     # labels of the batch
        sy = logit[yb, ar]
        d = torch.zeros_like(logit)
        d[logit > (sy - 1.0)] = -1.0                     # wrong-but-competitive: push down
        other = logit.clone()
        other[yb, ar] = -1e9
        safe = sy >= other.max(0).values + 1.0
        d[yb, ar] = torch.where(safe, 0.0, 1.0)          # true class: push up unless safe
        return d, logit

    # -- build --------------------------------------------------------------------------
    def build_sweep(self, y: torch.Tensor, bidx: torch.Tensor, n_build: int,
                    depth_pen: float = 1.0, usage_pen: float = 0.3, max_feats: int = 16384,
                    rebuild: bool = False, chunk: int = 8192) -> int:
        """Wire up to n_build slots this sweep. Normally fills empty slots; once the window is
        full and rebuild=True, it re-wires existing gate slots into (possibly deeper)
        compositions over the now-rich pool, so depth keeps growing with a fixed window. Each
        target is wired to a 2-input gate chosen by correlating pool signals with the slot's
        residual. Scales: we correlate only a capped, sampled feature set against just the slots
        wired this sweep, so cost is O(max_feats * n_build * Bb), not O(window^2)."""
        if self.n_gates_built >= self.max_gates:
            return 0
        empties = self.buildable[~self.filled[self.buildable]]
        if empties.numel() > 0:
            pool_t = empties
        elif rebuild:
            pool_t = self.buildable                                # re-wire gate slots to deepen
        else:
            return 0
        n = min(n_build, pool_t.numel(), self.max_gates - self.n_gates_built)
        tgt = pool_t[torch.randperm(pool_t.numel(), device=self.device)[:n]]  # slots to wire now
        d, _ = self.class_direction(y, bidx)

        # candidate input signals: sample max_feats filled slots from the path-length
        # distribution, weight ~ 1/(1+depth)^depth_pen / (1+usage)^usage_pen. Shallow, lightly-used
        # slots are picked most, so the correlation is computed against a mostly-shallow candidate
        # set; deep slots still appear sometimes. There is no Gumbel here -- the exploration comes
        # from this sampling plus the deliberately tiny build batch (noisy covariance), which lets
        # lower-covariance, not-yet-used inputs win on some sweeps.
        feats = self.filled.nonzero(as_tuple=False).flatten()
        w = torch.exp(-depth_pen * torch.log1p(self.depth[feats].to(torch.float32))
                      - usage_pen * torch.log1p(self.usage[feats].to(torch.float32)))
        if feats.numel() > max_feats:
            feats = feats[torch.multinomial(w, max_feats, replacement=False)]
        fb = gather_batch(self.win, feats, bidx)
        fb = fb - fb.mean(1, keepdim=True)

        # residual per target slot in {-1,0,+1}: fire more where its class should rise but it is
        # 0, fire less where its class should fall but it is 1.
        db = d[self.class_of[tgt]]
        v = gather_batch(self.win, tgt, bidx)
        r = torch.zeros_like(db)
        r[(db > 0) & (v == 0)] = 1.0
        r[(db < 0) & (v == 1)] = -1.0
        rc = r - r.mean(1, keepdim=True)

        # top-2 inputs per target by plain covariance over the sampled candidate set.
        best_val = torch.zeros((n, 2), device=self.device)
        best_fi = torch.zeros((n, 2), dtype=torch.long, device=self.device)
        for s in range(0, n, chunk):
            cov = fb @ rc[s:s + chunk].T                           # (K, |chunk|) correlation scores
            self_mask = feats[:, None] == tgt[s:s + chunk][None, :]
            key = cov.abs().masked_fill(self_mask, -1.0)           # never use a slot as its own input
            _, idx = key.topk(min(2, key.shape[0]), dim=0)
            best_fi[s:s + chunk] = idx.T
            best_val[s:s + chunk] = cov.gather(0, idx).T           # signed cov sets the NOT polarity

        a_slot = feats[best_fi[:, 0]]
        b_slot = feats[best_fi[:, 1]]
        pa = best_val[:, 0] >= 0                                   # NOT the input if corr is negative
        pb = best_val[:, 1] >= 0
        self._write(tgt, a_slot, b_slot, self._init_tt(pa, pb))
        return n

    def _init_tt(self, pa: torch.Tensor, pb: torch.Tensor) -> torch.Tensor:
        """AND/OR of two optionally-negated inputs as a (k,4) truth table [f00,f01,f10,f11]."""
        k = pa.shape[0]
        ar = torch.arange(k, device=self.device)
        ai, bi = pa.long(), pb.long()
        tt = torch.zeros((k, 4), dtype=torch.bool, device=self.device)
        if self.gate == "and":
            tt[ar, ai * 2 + bi] = True                             # fire only on the matching cell
        else:
            tt[:] = True
            tt[ar, (1 - ai) * 2 + (1 - bi)] = False                # off only where neither matches
        return tt

    def _write(self, slots: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
               tt: torch.Tensor) -> None:
        """Write gate outputs into the given slots; update scores, definitions and op history."""
        out = apply_gate(self.win[a], self.win[b], tt)             # (k, Wwords)
        old = unpack_bits(self.win[slots], self.D).to(torch.float32)
        new = unpack_bits(out, self.D).to(torch.float32)
        self.win[slots] = out
        self.score.index_add_(0, self.class_of[slots], new - old)
        self.def_a[slots] = a
        self.def_b[slots] = b
        self.def_tt[slots] = tt
        self.filled[slots] = True
        self.depth[slots] = torch.maximum(self.depth[a], self.depth[b]) + 1
        self.usage.index_add_(0, a, torch.ones_like(a))            # a, b now feed one more gate
        self.usage.index_add_(0, b, torch.ones_like(b))
        self.ops.append((slots.clone(), a.clone(), b.clone(), tt.clone()))
        self.n_gates_built += slots.numel()

    # -- coordinate descent (randomized: random gate, random bit, flip if it helps) ----
    def cd_pass(self, y: torch.Tensor, bidx: torch.Tensor, n_flip: int) -> int:
        """Pick n_flip random gate slots, pick one random truth-table bit on each, and flip the
        ones that lower the batch hinge loss. No scoring of the other bits, no sorting, no
        picking the best: just random candidate flips, kept only if they help. This keeps CD
        exploratory rather than greedily collapsing onto the locally-best move.

        The flips are evaluated against the same pre-pass logits and applied together. They are
        distinct, randomly chosen slots, so they barely interact (each flip only moves its own
        class score); the gain estimate per flip is exact in isolation."""
        gates = (self.filled & (torch.arange(self.WIN, device=self.device) >= self.N)
                 ).nonzero(as_tuple=False).flatten()
        if gates.numel() == 0:
            return 0
        # sample n_flip distinct gates, and a random bit (0..3) to try on each
        perm = torch.randperm(gates.numel(), device=self.device)[:min(n_flip, gates.numel())]
        cand = gates[perm]
        k_bit = torch.randint(4, (cand.shape[0],), device=self.device)

        bb = bidx.shape[0]
        ar = torch.arange(bb, device=self.device)
        yb = y[bidx]
        L = self.score[:, bidx] / self.tau                          # (C, Bb)
        sy = L[yb, ar]
        other = L.clone(); other[yb, ar] = -1e9
        m1, am1 = other.max(0)                                       # best competitor + its class
        other2 = other.clone(); other2[am1, ar] = -1e9
        m2 = other2.max(0).values                                   # 2nd best competitor

        g = cand.shape[0]
        gr = torch.arange(g, device=self.device)
        cg = self.class_of[cand]
        a_bit = gather_batch(self.win, self.def_a[cand], bidx)
        b_bit = gather_batch(self.win, self.def_b[cand], bidx)
        cell = (a_bit.long() * 2 + b_bit.long())                    # (G, Bb) tt entry hit per sample
        cur_tt = self.def_tt[cand]                                  # (G, 4)
        new_tt = cur_tt.clone()
        new_tt[gr, k_bit] = ~new_tt[gr, k_bit]                       # flip the one random bit
        cur_out = cur_tt.gather(1, cell).to(torch.float32)          # (G, Bb)
        new_out = new_tt.gather(1, cell).to(torch.float32)
        delta = (new_out - cur_out) / self.tau                      # change in this gate's vote

        # exact per-gate hinge delta: a flip only moves its own class score row
        is_true = cg[:, None] == yb[None, :]
        other_excl = torch.where(cg[:, None] == am1[None, :], m2[None, :], m1[None, :])
        Lc = L[cg]
        sy_new = torch.where(is_true, sy[None, :] + delta, sy[None, :])
        bo_new = torch.where(is_true, m1[None, :].expand(g, bb),
                             torch.maximum(other_excl, Lc + delta))
        base = torch.clamp(1.0 + m1 - sy, min=0).sum()
        loss = torch.clamp(1.0 + bo_new - sy_new, min=0).sum(1)     # (G,)
        keep = (base - loss) > 1e-6                                  # flip iff it helps
        if not keep.any():
            return 0

        sel = cand[keep]
        sel_tt = new_tt[keep]
        out = apply_gate(self.win[self.def_a[sel]], self.win[self.def_b[sel]], sel_tt)
        old = unpack_bits(self.win[sel], self.D).to(torch.float32)
        nw = unpack_bits(out, self.D).to(torch.float32)
        self.win[sel] = out
        self.score.index_add_(0, self.class_of[sel], nw - old)
        self.def_tt[sel] = sel_tt
        self.ops.append((sel.clone(), self.def_a[sel].clone(), self.def_b[sel].clone(),
                         sel_tt.clone()))
        return int(keep.sum().item())

    # -- inference ----------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, input_bits: torch.Tensor, y: torch.Tensor, batch: int = 8192) -> float:
        """Replay the op history on fresh images and return top-1 accuracy (%)."""
        d = input_bits.shape[1]
        correct = 0
        for i in range(0, d, batch):
            xb = input_bits[:, i:i + batch].to(self.device)
            ww = (xb.shape[1] + 63) // 64
            win = torch.zeros((self.WIN, ww), dtype=torch.int64, device=self.device)
            win[: self.N] = pack_bits(xb)
            for slots, a, b, tt in self.ops:                       # each batch reads earlier state
                win[slots] = apply_gate(win[a], win[b], tt)
            # sum window bits into class scores in slot chunks (never unpack the whole window)
            score = torch.zeros((self.C, xb.shape[1]), device=self.device)
            for s0 in range(0, self.WIN, 8192):
                sl = slice(s0, s0 + 8192)
                b = unpack_bits(win[sl], xb.shape[1]).to(torch.float32)
                score.index_add_(0, self.class_of[sl], b)
            pred = (score / self.tau).argmax(0).cpu()
            correct += (pred == y[i:i + batch]).sum().item()
        return 100.0 * correct / d

    # -- visualization snapshot ---------------------------------------------------------
    @torch.no_grad()
    def snapshot(self, grid: int = 180) -> dict:
        """Compact picture of the network right now, for the build animation:
        a depth map of the window, the histogram of which of the 16 boolean functions the gates
        became, and the depth distribution."""
        gate = self.filled & (torch.arange(self.WIN, device=self.device) >= self.N)
        tt = self.def_tt[gate]                                      # (G, 4) [f00,f01,f10,f11]
        fn_idx = (tt[:, 0].long() * 8 + tt[:, 1].long() * 4
                  + tt[:, 2].long() * 2 + tt[:, 3].long())          # 0..15
        fn = torch.bincount(fn_idx, minlength=16).cpu().numpy()
        dep = self.depth[self.filled]
        dcounts = torch.bincount(dep, minlength=10).cpu().numpy()
        # window depth map: -1 empty, 0 input bit, >=1 gate depth; max-pooled to a small image
        cols = int(math.ceil(self.WIN ** 0.5))
        rows = int(math.ceil(self.WIN / cols))
        val = torch.full((rows * cols,), -1.0, device=self.device)
        val[: self.WIN] = torch.where(self.filled, self.depth.float(), torch.tensor(-1.0, device=self.device))
        img = F.adaptive_max_pool2d(val.view(1, 1, rows, cols),
                                    (min(grid, rows), min(grid, cols)))[0, 0].cpu().numpy()
        return {"gates": int(gate.sum()), "fn": fn, "depth": dcounts, "img": img}


# ======================================================================================
# Encoding + training loop
# ======================================================================================
def encode(images: torch.Tensor, enc: Thermometer) -> torch.Tensor:
    """(D,3,32,32) -> (N, D) uint8 binary."""
    return enc(images).flatten(1).t().contiguous().to(torch.uint8)


def render_animation(snaps: list[dict], out_path: Path) -> None:
    """Animate the grown network: window depth map, the 16-function histogram, the depth
    distribution and the accuracy curve, one frame per recorded phase."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    phases = [s["phase"] for s in snaps]
    maxd = max(len(s["depth"]) for s in snaps)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    (a_img, a_fn), (a_dep, a_acc) = ax

    def update(i):
        s = snaps[i]
        for a in (a_img, a_fn, a_dep, a_acc):
            a.clear()
        im = a_img.imshow(s["img"], cmap="turbo", vmin=-1, vmax=max(2, maxd - 1))
        a_img.set_title(f"window  (slot depth; dark=empty)   gates={s['gates']:,}")
        a_img.set_xticks([]); a_img.set_yticks([])

        a_fn.bar(range(16), s["fn"], color="tab:blue")
        a_fn.set_xticks(range(16)); a_fn.set_xticklabels(FN_NAMES, rotation=90, fontsize=8)
        a_fn.set_title("which of the 16 gates"); a_fn.set_ylabel("count")

        dd = s["depth"]
        a_dep.bar(range(len(dd)), dd, color="tab:green")
        a_dep.set_title("slot depth distribution"); a_dep.set_xlabel("depth")
        a_dep.set_ylabel("slots")

        j = i + 1
        a_acc.plot(phases[:j], [t["tr"] for t in snaps[:j]], label="train", color="tab:blue")
        a_acc.plot(phases[:j], [t["va"] for t in snaps[:j]], label="val", color="tab:orange")
        a_acc.plot(phases[:j], [t["te"] for t in snaps[:j]], label="test", color="tab:green")
        a_acc.set_xlim(0, max(phases)); a_acc.set_ylim(0, max(60, max(t["te"] for t in snaps) + 5))
        a_acc.set_title(f"accuracy   phase {s['phase']}   test={s['te']:.1f}%")
        a_acc.set_xlabel("phase"); a_acc.legend(loc="lower right"); a_acc.grid(alpha=0.3)
        fig.suptitle("backprop-free LUT network: greedy build + coordinate descent", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        return []

    anim = FuncAnimation(fig, update, frames=len(snaps), blit=False)
    anim.save(str(out_path), writer=PillowWriter(fps=3))
    plt.close(fig)
    print(f"wrote animation: {out_path}  ({len(snaps)} frames)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/cifar-10-batches-py"))
    p.add_argument("--download", action="store_true")
    p.add_argument("--train-size", type=int, default=0, help="train+val pool size (0=full 50k)")
    # N = in_channels * H * W * num_bits = 3*32*32*b = 3072*b, and WIN = f*N must divide by C=10,
    # i.e. b*f must be a multiple of 5. The defaults b=5, f=2 satisfy this exactly (N=15360,
    # WIN=30720, H=WIN/C=3072). The code trims WIN to a multiple of C anyway, as a safety net.
    p.add_argument("--num-bits", type=int, default=5, help="thermometer bits per channel (b)")
    p.add_argument("--window-factor", type=float, default=2.0, help="window = f * N slots")
    p.add_argument("--max-gates", type=int, default=200000, help="upper limit on gates built")
    p.add_argument("--max-feats", type=int, default=16384,
                   help="cap on pool signals correlated per build sweep (bounds the matmul)")
    p.add_argument("--gate", choices=["and", "or"], default="and", help="initial gate family")
    p.add_argument("--depth-penalty", type=float, default=1.0,
                   help="sharpness of the inverse-path-length input sampling (weight ~ 1/(1+depth)^this)")
    p.add_argument("--usage-penalty", type=float, default=0.3,
                   help="bias against reusing already heavily-used slots as inputs")
    p.add_argument("--rebuild", action="store_true",
                   help="after the window is full, keep re-wiring slots into deeper gates "
                        "(depth grows with a fixed window) until --max-gates build-ops")
    p.add_argument("--build-batch", type=int, default=64,
                   help="samples per build correlation (small = noisy cov = exploration)")
    p.add_argument("--cd-batch", type=int, default=16384, help="samples per CD pass (large = less overfit)")
    p.add_argument("--cd-flips", type=int, default=8192,
                   help="random (gate,bit) flips attempted per CD call; kept only if they help")
    p.add_argument("--build-per-phase", type=int, default=10000,
                   help="empty slots filled in each build phase")
    p.add_argument("--cd-fraction", type=float, default=0.25,
                   help="CD bitflips per phase as a fraction of all gates built so far")
    p.add_argument("--extra-cd-phases", type=int, default=30,
                   help="CD-only phases after the window is full")
    p.add_argument("--eval-every", type=int, default=3, help="phases between evals")
    p.add_argument("--viz", action="store_true", help="record a per-phase build animation")
    p.add_argument("--viz-out", type=Path, default=Path("scratch/grow_anim.gif"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dev = args.device
    print(f"device={dev}  args={vars(args)}", flush=True)

    train_x, train_y, test_x, test_y = load_cifar10(args.data_dir, args.download)
    work_x, work_y = train_x, train_y
    if args.train_size > 0:
        work_x, work_y = work_x[: args.train_size], work_y[: args.train_size]
    n_val = max(1, round(len(work_x) * 0.1))
    val_x, val_y = work_x[-n_val:], work_y[-n_val:]
    pool_x, pool_y = work_x[:-n_val], work_y[:-n_val]
    print(f"train={len(pool_x)} val={len(val_x)} test={len(test_x)}", flush=True)

    enc = Thermometer(num_bits=args.num_bits).fit(pool_x[:2000])
    Xtr, Xva, Xte = encode(pool_x, enc), encode(val_x, enc), encode(test_x, enc)
    n_inputs = Xtr.shape[0]
    ytr = pool_y.to(dev)

    circ = GrownCircuit(n_inputs, 10, args.window_factor, gate=args.gate,
                        max_gates=args.max_gates, device=dev)
    circ.set_inputs(Xtr)
    exact = (n_inputs * args.window_factor) % 10 == 0
    print(f"N=C*H*W*b={n_inputs}  WIN=f*N={circ.WIN}  H=WIN/C={circ.H}  H*C={circ.H * 10}  "
          f"tau={circ.tau:.1f}  exact={exact}  empty slots to fill={circ.WIN - n_inputs}",
          flush=True)

    def rbatch(n: int) -> torch.Tensor:
        return torch.randint(circ.D, (min(n, circ.D),), device=dev)

    t0 = time.time()
    header = "  tag |  gates  | build |   cd   | tr_acc | va_acc | te_acc |  time"
    print("\n" + header + "\n" + "-" * len(header), flush=True)

    best = {"va": -1.0, "te": -1.0, "phase": 0, "gates": 0}

    def show(tag: str, built: int, flips: int) -> tuple[float, float, float]:
        tr = circ.evaluate(Xtr, pool_y)
        va = circ.evaluate(Xva, val_y)
        te = circ.evaluate(Xte, test_y)
        print(f"{tag:>6} | {circ.n_gates_built:7d} | {built:5d} | {flips:6d} | "
              f"{tr:6.2f} | {va:6.2f} | {te:6.2f} | {time.time() - t0:4.0f}s", flush=True)
        if va > best["va"]:
            best.update(va=va, te=te, phase=int(tag[1:]) if tag[1:].isdigit() else 0,
                        gates=circ.n_gates_built)
        return tr, va, te

    show("base", 0, 0)  # inputs only, before any gate
    snaps: list[dict] = []

    def record(phase: int) -> None:
        if not args.viz:
            return
        s = circ.snapshot()
        s.update(phase=phase, tr=circ.evaluate(Xtr, pool_y), va=circ.evaluate(Xva, val_y),
                 te=circ.evaluate(Xte, test_y))
        snaps.append(s)

    # Each phase: build a chunk of gates, then run CD bitflips equal to a fraction of all gates
    # built so far. Build is constant per phase while the gate count grows, so CD automatically
    # takes over as the window fills (more CD vs build over time). The first build covers every
    # class (round-robin), so each GroupSum already has gates before the first CD.
    def cd_phase() -> int:
        target = int(round(args.cd_fraction * circ.n_gates_built))
        done = 0
        while done < target:
            nf = min(args.cd_flips, target - done)
            circ.cd_pass(ytr, rbatch(args.cd_batch), nf)
            done += nf
        return done

    phase, full_at = 0, None
    while True:
        phase += 1
        built = circ.build_sweep(ytr, rbatch(args.build_batch), args.build_per_phase,
                                 depth_pen=args.depth_penalty, usage_pen=args.usage_penalty,
                                 max_feats=args.max_feats, rebuild=args.rebuild)
        flips = cd_phase()
        full = not (~circ.filled[circ.buildable]).any()
        if full and full_at is None:
            full_at = phase
        if phase % args.eval_every == 0 or (full and full_at == phase):
            show(f"p{phase}", built, flips)
        record(phase)
        if circ.n_gates_built >= args.max_gates:
            break
        if not args.rebuild and full_at is not None and phase >= full_at + args.extra_cd_phases:
            break

    tr, va, te = (circ.evaluate(Xtr, pool_y), circ.evaluate(Xva, val_y),
                  circ.evaluate(Xte, test_y))
    gd = circ.depth[circ.filled]
    print(f"\nFINAL  gates={circ.n_gates_built}  ops={len(circ.ops)}  "
          f"train={tr:.2f}  val={va:.2f}  test={te:.2f}", flush=True)
    print(f"depth: mean={gd.float().mean():.2f}  max={int(gd.max())}  "
          f"usage: mean={circ.usage[circ.filled].float().mean():.2f}  "
          f"max={int(circ.usage.max())}", flush=True)
    print(f"BEST  val={best['va']:.2f}  test={best['te']:.2f}  "
          f"at phase {best['phase']}  gates={best['gates']}", flush=True)

    if args.viz and snaps:
        args.viz_out.parent.mkdir(parents=True, exist_ok=True)
        render_animation(snaps, args.viz_out)


if __name__ == "__main__":
    main()
