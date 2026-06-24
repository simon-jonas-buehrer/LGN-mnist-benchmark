"""Backprop-free LUT network grown by greedy correlation + coordinate descent.

A scratch experiment: build the same kind of boolean-gate circuit as model.py, but with no
gradients at all. Everything is binary and bitpacked, so building and inference are a few GPU
matmuls and bit ops.

The window
----------
The whole model is one fixed window of WIN = f * N slots, which is also the entire head. It
starts as f tiled copies of the N thermometer bits (input j's f copies are spread over f
different classes (j+k)%C, so no class group holds the same input twice). The GroupSum head
sums the window by class (class = slot % C, H = WIN / C slots per class, score divided by
tau = sqrt(H); tau is equal across classes so it does not affect the argmax). The classifier is
a vote: the class with the most firing slots wins.

Gates: k-input LUTs
-------------------
Each slot is a K-input lookup table: it reads K signals and stores a 2**K-bit truth table, so it
can express any of 2**(2**K) functions of its inputs. CD reaches all of them and tends to
distribute gates across the function space (it does not collapse to AND -- the seed AND is only a
starting point). The reason to raise K is capacity/selectivity: with more inputs a single LUT can
carve a much finer region of input space, which is what lets the popcount head separate (and fit)
the training set. K is the main knob here (--fan-in).

Build (no backprop)
-------------------
Every slot is buildable. Each phase, on a tiny random batch, we *replace* a chunk of random
slots with fresh gates:

  1. residual per target slot in {-1,0,+1} from the multiclass-hinge subgradient: fire more
     where its class should go up but the slot is 0, fire less where its class should go down but
     it is 1.
  2. sample candidate input signals from the inverse-path-length distribution (weight
     ~ 1/(1+depth)^depth_pen / (1+usage)^usage_pen, so shallow lightly-used slots are preferred)
     and correlate them with the residual (covariance over the batch).
  3. wire the top-K by covariance into an AND of the K (optionally NOT-ed) inputs. Exploration
     comes from the tiny build batch (noisy covariance) plus the sampling. CD later reaches every
     2**K-bit function.

Building both forms and deepens the circuit; once every copy has been rebuilt no raw inputs
remain in the pool.

Coordinate descent
------------------
Randomized: pick random gate slots, pick one random truth-table bit on each, flip iff it lowers
the batch hinge loss (closed form, since flipping a slot only moves its own class score). No
sorting, no picking the best. Each phase runs cd_fraction * WIN bitflips; an optional long final
CD-only phase polishes at the end.

Inference replays the ordered op-history (vectorized, exact) on fresh images.

Run
---
    .venv/bin/python scratch/grow_lut.py --device cuda --train-size 0 --window-factor 8 --fan-in 4
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
    chunks and cast to uint8 immediately so the int64 (n, chunk, 64) temporary stays small."""
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


def apply_gate(ins: torch.Tensor, tt: torch.Tensor) -> torch.Tensor:
    """Packed K-input LUT. ins: (n, K, W) int64 signal words; tt: (n, 2**K) bool truth table
    indexed by cell = sum_i input_i_bit * 2**i. Returns (n, W) int64.

    out fires on word position p iff tt[c] is set for the cell c selected by the K input bits at
    p; we OR together, over the cells tt has set, the AND of (input_i if cell bit i else ~input_i)."""
    n, k, w = ins.shape
    out = ins.new_zeros((n, w))
    ones = ins.new_full((n, w), -1)                                # all-ones mask
    for c in range(tt.shape[1]):                                   # 2**K cells
        m = ones
        for i in range(k):
            m = m & (ins[:, i] if (c >> i) & 1 else ~ins[:, i])
        out = out | torch.where(tt[:, c:c + 1], m, ins.new_zeros(()))
    return out


def apply_conj(ins: torch.Tensor, params: torch.Tensor, k: int, terms: int) -> torch.Tensor:
    """Packed K-input C-term DNF gate. ins: (n, K, W) int64; params: (n, terms*2K) bool, each term
    is (active?, polarity) per input. Returns (n, W) = OR over terms of (AND over a term's active
    inputs of its literal). An empty term (no active inputs) contributes 0. terms*2K bits."""
    n, _, w = ins.shape
    out = ins.new_zeros((n, w))
    for t in range(terms):
        b = t * 2 * k
        act, pol = params[:, b:b + k], params[:, b + k:b + 2 * k]
        term = ins.new_full((n, w), -1)                            # all ones
        for i in range(k):
            lit = torch.where(pol[:, i:i + 1], ins[:, i], ~ins[:, i])
            term = torch.where(act[:, i:i + 1], term & lit, term)
        term = torch.where(act.any(1, keepdim=True), term, ins.new_zeros(()))  # empty term -> 0
        out = out | term
    return out


# ======================================================================================
# The grown circuit
# ======================================================================================
class GrownCircuit:
    def __init__(self, n_inputs: int, n_classes: int, window_factor: float,
                 *, fan_in: int, gate_type: str, terms: int = 1, max_gates: int, device: str):
        self.N = n_inputs
        self.C = n_classes
        self.K = fan_in
        self.gate_type = gate_type                                 # "lut" or "conj"
        self.terms = terms                                         # DNF terms per conj gate
        self.TT = 1 << fan_in                                      # truth-table length 2**K
        # Per-gate parameter bits: a full LUT is 2**K; a conj gate is an OR of `terms` conjunctions,
        # each 2K bits (per input: active? and polarity), so it stores terms*2K -- linear in K. A
        # full LUT is the full DNF (up to 2**K minterms); conj keeps only `terms` of them.
        self.P = self.TT if gate_type == "lut" else terms * 2 * fan_in
        win = int(round(window_factor * n_inputs))
        win -= win % n_classes                                     # clean multiple of C
        self.WIN = win
        self.H = win // n_classes
        self.tau = math.sqrt(self.H)
        self.max_gates = max_gates
        self.device = device
        self.pw = (1 << torch.arange(fan_in, device=device)).long()   # 2**i, for LUT cell indexing

        # The window is WIN = f*N slots, the whole head. It starts as f tiled copies of the N
        # encoding bits; every slot is buildable, building replaces a slot (an input copy or a
        # gate) with a fresh K-input gate.
        self.class_of = torch.arange(win, device=device) % n_classes
        self.depth = torch.zeros(win, dtype=torch.long, device=device)
        self.usage = torch.zeros(win, dtype=torch.long, device=device)
        self.def_in = torch.full((win, fan_in), -1, dtype=torch.long, device=device)  # -1 = copy
        self.def_p = torch.zeros((win, self.P), dtype=torch.bool, device=device)   # gate params
        self.buildable = torch.arange(win, device=device)

        self.ops: list[tuple] = []                   # ordered op-batches (slots, ins, tt)
        self.n_gates_built = 0
        self.win = None                              # (WIN, Wwords) int64 signal table over train
        self.score = None                            # (C, D) class scores over the train set

    # -- setup --------------------------------------------------------------------------
    def set_inputs(self, input_bits: torch.Tensor) -> None:
        """input_bits: (N, D) uint8. Tile f copies across the window; the head sums all of it."""
        d = input_bits.shape[1]
        self.D = d
        self.Wwords = (d + 63) // 64
        inp = pack_bits(input_bits.to(self.device))                   # (N, Wwords)
        self.tile = self._tiling()                                    # slot -> input index
        self.win = inp[self.tile].contiguous()                        # (WIN, Wwords) f copies
        self._recompute_score()

    def _tiling(self) -> torch.Tensor:
        """Which input bit each window slot starts as. Distribute the f copies of input j over f
        *different* classes (c = (j+k)%C), so no class group holds the same input twice (requires
        f<=C and N divisible by C). Else fall back to slot % N."""
        dev = self.device
        f = self.WIN // self.N
        tile = torch.arange(self.WIN, device=dev) % self.N
        if f * self.N != self.WIN or f > self.C or self.N % self.C != 0:
            return tile
        j = torch.arange(self.N, device=dev).repeat(f)
        k = torch.arange(f, device=dev).repeat_interleave(self.N)
        cls = (j + k) % self.C
        inp_sorted = j[torch.argsort(cls, stable=True)]
        h = self.WIN // self.C
        idx = torch.arange(self.WIN, device=dev)
        slot = (idx // h) + (idx % h) * self.C
        tile[slot] = inp_sorted
        return tile

    def _recompute_score(self) -> None:
        """Class scores = popcount of each class's window slots, summed in slot chunks."""
        self.score = torch.zeros((self.C, self.D), dtype=torch.float32, device=self.device)
        for s0 in range(0, self.WIN, 8192):
            sl = slice(s0, min(s0 + 8192, self.WIN))
            b = unpack_bits(self.win[sl], self.D).to(torch.float32)
            self.score.index_add_(0, self.class_of[sl], b)

    # -- supervision --------------------------------------------------------------------
    def class_direction(self, y: torch.Tensor, bidx: torch.Tensor):
        """Per-class desired direction in {-1,0,+1} (multiclass-hinge subgradient)."""
        logit = self.score[:, bidx] / self.tau          # (C, Bb)
        bb = bidx.shape[0]
        ar = torch.arange(bb, device=self.device)
        yb = y[bidx]
        sy = logit[yb, ar]
        d = torch.zeros_like(logit)
        d[logit > (sy - 1.0)] = -1.0                     # wrong-but-competitive: push down
        other = logit.clone()
        other[yb, ar] = -1e9
        safe = sy >= other.max(0).values + 1.0
        d[yb, ar] = torch.where(safe, 0.0, 1.0)          # true class: push up unless safe
        return d, logit

    # -- gate dispatch (lut truth table vs conj shared-AND weights) ----------------------
    def apply_full(self, ins_words: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Gate output over all D: ins_words (n, K, W), params (n, P) -> (n, W)."""
        if self.gate_type == "lut":
            return apply_gate(ins_words, params)
        return apply_conj(ins_words, params, self.K, self.terms)

    def apply_batch(self, ins_slots: torch.Tensor, params: torch.Tensor,
                    bidx: torch.Tensor) -> torch.Tensor:
        """Gate output on a batch: ins_slots (G, K) signal indices, params (G, P) -> (G, Bb)."""
        g, k = ins_slots.shape
        bits = gather_batch(self.win, ins_slots.reshape(-1), bidx).view(g, k, -1)   # (G,K,Bb)
        if self.gate_type == "lut":
            cell = (bits.long() * self.pw.view(1, k, 1)).sum(1)    # (G, Bb)
            return params.gather(1, cell).to(torch.float32)
        out = torch.zeros((g, bits.shape[2]), device=self.device)  # OR over DNF terms
        for t in range(self.terms):
            b = t * 2 * k
            act, pol = params[:, b:b + k, None], params[:, b + k:b + 2 * k, None]
            lit = torch.where(pol, bits, 1.0 - bits)
            term = torch.where(act, lit, torch.ones_like(bits)).prod(1)   # AND of active
            term = torch.where(act.squeeze(-1).any(1, keepdim=True), term, torch.zeros_like(term))
            out = torch.maximum(out, term)
        return out

    # -- build --------------------------------------------------------------------------
    def build_sweep(self, y: torch.Tensor, bidx: torch.Tensor, n_build: int,
                    depth_pen: float = 2.0, usage_pen: float = 0.3, max_feats: int = 16384,
                    chunk: int = 8192) -> int:
        """Replace up to n_build random slots with fresh K-input gates whose K inputs are the
        top-K signals by covariance with the slot's residual."""
        if self.n_gates_built >= self.max_gates:
            return 0
        pool_t = self.buildable
        n = min(n_build, pool_t.numel(), self.max_gates - self.n_gates_built)
        tgt = pool_t[torch.randperm(pool_t.numel(), device=self.device)[:n]]
        d, _ = self.class_direction(y, bidx)

        # candidate input signals: sample max_feats slots from the inverse-path-length distribution
        feats = self.buildable
        w = torch.exp(-depth_pen * torch.log1p(self.depth[feats].to(torch.float32))
                      - usage_pen * torch.log1p(self.usage[feats].to(torch.float32)))
        if feats.numel() > max_feats:
            feats = feats[torch.multinomial(w, max_feats, replacement=False)]
        fb = gather_batch(self.win, feats, bidx)
        fb = fb - fb.mean(1, keepdim=True)

        # Process the targets in chunks: residual -> covariance -> top-K -> write, per chunk. This
        # bounds memory (the per-chunk _write unpacks only (chunk, D)), so build_per_phase can be
        # huge without OOM.
        kk = min(self.K, feats.numel())
        for s in range(0, n, chunk):
            tc = tgt[s:s + chunk]
            db = d[self.class_of[tc]]
            v = gather_batch(self.win, tc, bidx)
            r = torch.zeros_like(db)
            r[(db > 0) & (v == 0)] = 1.0
            r[(db < 0) & (v == 1)] = -1.0
            rc = r - r.mean(1, keepdim=True)
            cov = fb @ rc.T                                        # (feats, |chunk|)
            cov = cov.masked_fill(feats[:, None] == tc[None, :], 0.0)   # never use a slot as own input
            _, idx = cov.abs().topk(kk, dim=0)                     # (kk, |chunk|)
            ins = feats[idx.T]                                     # (|chunk|, kk)
            pol = cov.gather(0, idx).T >= 0
            if kk < self.K:                                        # tiny pool: pad
                ins = torch.cat([ins, ins[:, :1].expand(ins.shape[0], self.K - kk)], dim=1)
                pol = torch.cat([pol, pol[:, :1].expand(pol.shape[0], self.K - kk)], dim=1)
            self._write(tc, ins, self._init_params(pol))
        return n

    def _init_params(self, pol: torch.Tensor) -> torch.Tensor:
        """Seed gate = AND of the K inputs at their wanted polarity. lut: one-hot truth table at
        that cell. conj: all inputs active, polarity from pol."""
        n = pol.shape[0]
        if self.gate_type == "lut":
            cell = (pol.long() * self.pw).sum(1)                   # (n,)
            p = torch.zeros((n, self.P), dtype=torch.bool, device=self.device)
            p[torch.arange(n, device=self.device), cell] = True
            return p
        # conj: seed term 0 = AND of all K inputs at their polarity; other terms empty (output 0)
        p = torch.zeros((n, self.P), dtype=torch.bool, device=self.device)
        p[:, :self.K] = True                                       # term 0 active
        p[:, self.K:2 * self.K] = pol                             # term 0 polarity
        return p

    def _write(self, slots: torch.Tensor, ins: torch.Tensor, params: torch.Tensor) -> None:
        """Write K-input gate outputs into the slots; update scores, definitions and op history."""
        out = self.apply_full(self.win[ins], params)               # (n, Wwords)
        old = unpack_bits(self.win[slots], self.D).to(torch.float32)
        new = unpack_bits(out, self.D).to(torch.float32)
        self.win[slots] = out
        self.score.index_add_(0, self.class_of[slots], new - old)
        self.def_in[slots] = ins
        self.def_p[slots] = params
        self.depth[slots] = self.depth[ins].max(1).values + 1
        self.usage.index_add_(0, ins.reshape(-1), torch.ones(ins.numel(), dtype=torch.long,
                                                              device=self.device))
        self.ops.append((slots.clone(), ins.clone(), params.clone()))
        self.n_gates_built += slots.numel()

    # -- coordinate descent (randomized: random gate, random bit, flip if it helps) ----
    def cd_pass(self, y: torch.Tensor, bidx: torch.Tensor, n_flip: int) -> int:
        gates = (self.def_in[:, 0] >= 0).nonzero(as_tuple=False).flatten()   # built gates
        if gates.numel() == 0:
            return 0
        perm = torch.randperm(gates.numel(), device=self.device)[:min(n_flip, gates.numel())]
        cand = gates[perm]
        k_bit = torch.randint(self.P, (cand.shape[0],), device=self.device)  # random param bit to flip

        bb = bidx.shape[0]
        ar = torch.arange(bb, device=self.device)
        yb = y[bidx]
        L = self.score[:, bidx] / self.tau
        sy = L[yb, ar]
        other = L.clone(); other[yb, ar] = -1e9
        m1, am1 = other.max(0)
        other2 = other.clone(); other2[am1, ar] = -1e9
        m2 = other2.max(0).values

        g = cand.shape[0]
        gr = torch.arange(g, device=self.device)
        cg = self.class_of[cand]
        cur_out = gather_batch(self.win, cand, bidx)               # (G, Bb) current gate output
        new_p = self.def_p[cand].clone()
        new_p[gr, k_bit] = ~new_p[gr, k_bit]                       # flip one param bit
        new_out = self.apply_batch(self.def_in[cand], new_p, bidx)  # gate output with the flip
        delta = (new_out - cur_out) / self.tau

        is_true = cg[:, None] == yb[None, :]
        other_excl = torch.where(cg[:, None] == am1[None, :], m2[None, :], m1[None, :])
        Lc = L[cg]
        sy_new = torch.where(is_true, sy[None, :] + delta, sy[None, :])
        bo_new = torch.where(is_true, m1[None, :].expand(g, bb),
                             torch.maximum(other_excl, Lc + delta))
        base = torch.clamp(1.0 + m1 - sy, min=0).sum()
        loss = torch.clamp(1.0 + bo_new - sy_new, min=0).sum(1)
        keep = (base - loss) > 1e-6
        if not keep.any():
            return 0

        sel = cand[keep]
        sel_p = new_p[keep]
        out = self.apply_full(self.win[self.def_in[sel]], sel_p)
        old = unpack_bits(self.win[sel], self.D).to(torch.float32)
        nw = unpack_bits(out, self.D).to(torch.float32)
        self.win[sel] = out
        self.score.index_add_(0, self.class_of[sel], nw - old)
        self.def_p[sel] = sel_p
        self.ops.append((sel.clone(), self.def_in[sel].clone(), sel_p.clone()))
        return int(keep.sum().item())

    # -- inference ----------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, input_bits: torch.Tensor, y: torch.Tensor, batch: int = 8192) -> float:
        """Replay the op history on fresh images and return top-1 accuracy (%)."""
        d = input_bits.shape[1]
        correct = 0
        for i in range(0, d, batch):
            xb = input_bits[:, i:i + batch].to(self.device)
            win = pack_bits(xb)[self.tile].contiguous()            # f tiled copies of the encoding
            for slots, ins, params in self.ops:                    # each batch reads earlier state
                win[slots] = self.apply_full(win[ins], params)
            score = torch.zeros((self.C, xb.shape[1]), device=self.device)
            for s0 in range(0, self.WIN, 8192):
                sl = slice(s0, min(s0 + 8192, self.WIN))
                b = unpack_bits(win[sl], xb.shape[1]).to(torch.float32)
                score.index_add_(0, self.class_of[sl], b)
            pred = (score / self.tau).argmax(0).cpu()
            correct += (pred == y[i:i + batch]).sum().item()
        return 100.0 * correct / d

    # -- visualization snapshot ---------------------------------------------------------
    @torch.no_grad()
    def snapshot(self, grid: int = 180) -> dict:
        """Window depth map, the truth-table popcount histogram (how specific the gates are), and
        the depth distribution."""
        gate = self.def_in[:, 0] >= 0                              # built gates (not raw copies)
        pop = self.def_p[gate].sum(1)                              # param bits set per gate
        ttpop = torch.bincount(pop, minlength=min(self.P, 64) + 1).cpu().numpy()
        dcounts = torch.bincount(self.depth, minlength=10).cpu().numpy()
        cols = int(math.ceil(self.WIN ** 0.5))
        rows = int(math.ceil(self.WIN / cols))
        val = torch.full((rows * cols,), -1.0, device=self.device)
        val[: self.WIN] = self.depth.float()                       # 0 = raw copy, >=1 = gate depth
        img = F.adaptive_max_pool2d(val.view(1, 1, rows, cols),
                                    (min(grid, rows), min(grid, cols)))[0, 0].cpu().numpy()
        return {"gates": int(gate.sum()), "ttpop": ttpop, "depth": dcounts, "img": img}


# ======================================================================================
# Encoding + training loop
# ======================================================================================
def encode(images: torch.Tensor, enc: Thermometer) -> torch.Tensor:
    """(D,3,32,32) -> (N, D) uint8 binary via the thermometer encoder."""
    return enc(images).flatten(1).t().contiguous().to(torch.uint8)


def render_animation(snaps: list[dict], out_path: Path) -> None:
    """Animate: window depth map, truth-table popcount histogram, depth distribution, accuracy
    curve (+ total-gates curve on a second axis), one frame per recorded phase."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    phases = [s["phase"] for s in snaps]
    maxd = max(len(s["depth"]) for s in snaps)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    (a_img, a_fn), (a_dep, a_acc) = ax
    a_g = a_acc.twinx()

    def update(i):
        s = snaps[i]
        for a in (a_img, a_fn, a_dep, a_acc, a_g):
            a.clear()
        a_img.imshow(s["img"], cmap="turbo", vmin=-1, vmax=max(2, maxd - 1))
        a_img.set_title(f"window  (slot depth; dark=copy)   gates={s['gates']:,}")
        a_img.set_xticks([]); a_img.set_yticks([])

        a_fn.bar(range(len(s["ttpop"])), s["ttpop"], color="tab:blue")
        a_fn.set_title("gate parameter bits set  (the LUT weights)")
        a_fn.set_xlabel("# param bits set per gate"); a_fn.set_ylabel("gates")

        dd = s["depth"]
        a_dep.bar(range(len(dd)), dd, color="tab:green")
        a_dep.set_title("slot depth distribution"); a_dep.set_xlabel("depth"); a_dep.set_ylabel("slots")

        j = i + 1
        a_acc.plot(phases[:j], [t["tr"] for t in snaps[:j]], label="train", color="tab:blue")
        a_acc.plot(phases[:j], [t["va"] for t in snaps[:j]], label="val", color="tab:orange")
        a_acc.plot(phases[:j], [t["te"] for t in snaps[:j]], label="test", color="tab:green")
        a_acc.set_xlim(0, max(phases)); a_acc.set_ylim(0, max(60, max(t["te"] for t in snaps) + 5))
        a_acc.set_title(f"accuracy   phase {s['phase']}   test={s['te']:.1f}%")
        a_acc.set_xlabel("phase"); a_acc.set_ylabel("accuracy (%)")
        a_acc.legend(loc="lower right"); a_acc.grid(alpha=0.3)
        # second y-axis (log): the two schedules -- cumulative gates built and cumulative flips
        a_g.plot(phases[:j], [max(1, t["n_built"]) for t in snaps[:j]], color="tab:red",
                 ls="--", lw=1, label="gates built")
        a_g.plot(phases[:j], [max(1, t.get("n_flipped", 0)) for t in snaps[:j]], color="tab:purple",
                 ls=":", lw=1, label="bitflips")
        a_g.set_yscale("log")
        a_g.set_ylim(1, max(10, max(max(t["n_built"], t.get("n_flipped", 0)) for t in snaps)))
        a_g.set_ylabel("cumulative (log): gates built / bitflips")
        a_g.legend(loc="upper left", fontsize=7)
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
    # N = 3*32*32*b = 3072*b; WIN = f*N must divide by C=10, so b*f must be a multiple of 5.
    p.add_argument("--num-bits", type=int, default=5, help="thermometer bits per channel (b)")
    p.add_argument("--window-factor", type=float, default=8.0, help="window = f * N slots")
    p.add_argument("--fan-in", type=int, default=4, help="inputs per gate K")
    p.add_argument("--gate-type", choices=["lut", "conj"], default="lut",
                   help="lut: full 2**K truth table. conj: DNF of `terms` conjunctions (scales O(K))")
    p.add_argument("--terms", type=int, default=1,
                   help="conj only: DNF terms per gate (each terms*2K bits); 1 = single conjunction")
    p.add_argument("--max-gates", type=int, default=400000, help="upper limit on gates built")
    p.add_argument("--phases", type=int, default=0,
                   help="if >0, run exactly this many phases and ramp the schedule by phase "
                        "progress (lets build ramp to 0 while CD keeps going); else stop at max-gates")
    p.add_argument("--max-feats", type=int, default=16384,
                   help="cap on pool signals correlated per build sweep (bounds the matmul)")
    p.add_argument("--depth-penalty", type=float, default=2.0,
                   help="sharpness of the inverse-path-length input sampling (~ 1/(1+depth)^this)")
    p.add_argument("--usage-penalty", type=float, default=0.3,
                   help="bias against reusing already heavily-used slots as inputs")
    p.add_argument("--build-batch", type=int, default=64,
                   help="samples per build correlation (small = noisy cov = exploration)")
    p.add_argument("--cd-batch", type=int, default=8192, help="samples per CD pass")
    p.add_argument("--cd-flips", type=int, default=1024,
                   help="random (gate,bit) flips attempted per CD call; kept only if they help")
    # Per-phase schedule: gates (re)built and CD bitflips both ramp linearly with progress
    # (n_gates_built / max_gates), so you can build-heavy early and flip-heavy late.
    p.add_argument("--build-start", type=int, default=2000, help="slots (re)built in the first phase")
    p.add_argument("--build-end", type=int, default=2000, help="slots (re)built in the last phase")
    p.add_argument("--cd-start", type=int, default=60000, help="CD bitflips in the first phase")
    p.add_argument("--cd-end", type=int, default=60000, help="CD bitflips in the last phase")
    p.add_argument("--final-cd-flips", type=int, default=0,
                   help="long CD-only annealing after the gate budget: total random bitflips")
    p.add_argument("--eval-every", type=int, default=10, help="phases between evals")
    p.add_argument("--viz", action="store_true", help="record a build animation")
    p.add_argument("--viz-every", type=int, default=1, help="phases between animation frames")
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

    circ = GrownCircuit(n_inputs, 10, args.window_factor, fan_in=args.fan_in,
                        gate_type=args.gate_type, terms=args.terms,
                        max_gates=args.max_gates, device=dev)
    circ.set_inputs(Xtr)
    exact = (n_inputs * args.window_factor) % 10 == 0
    print(f"N={n_inputs}  WIN=f*N={circ.WIN}  H={circ.H}  K={circ.K}  gate={circ.gate_type}  "
          f"P(bits/gate)={circ.P}  tau={circ.tau:.1f}  exact={exact}", flush=True)

    def rbatch(n: int) -> torch.Tensor:
        return torch.randint(circ.D, (min(n, circ.D),), device=dev)

    t0 = time.time()
    header = "  tag |  gates  | build |   cd   | tr_acc | va_acc | te_acc |  time"
    print("\n" + header + "\n" + "-" * len(header), flush=True)

    best = {"va": -1.0, "te": -1.0, "tag": "", "gates": 0}

    def show(tag: str, built: int, flips: int) -> tuple[float, float, float]:
        tr = circ.evaluate(Xtr, pool_y)
        va = circ.evaluate(Xva, val_y)
        te = circ.evaluate(Xte, test_y)
        print(f"{tag:>6} | {circ.n_gates_built:7d} | {built:5d} | {flips:6d} | "
              f"{tr:6.2f} | {va:6.2f} | {te:6.2f} | {time.time() - t0:4.0f}s", flush=True)
        if va > best["va"]:
            best.update(va=va, te=te, tag=tag, gates=circ.n_gates_built)
        return tr, va, te

    show("base", 0, 0)  # tiled inputs, before any gate
    snaps: list[dict] = []

    flips_total = [0]  # cumulative CD bitflips attempted (the CD schedule, for the plot)

    def record(phase: int) -> None:
        if not args.viz or phase % args.viz_every != 0:
            return
        s = circ.snapshot()
        s.update(phase=phase, n_built=circ.n_gates_built, n_flipped=flips_total[0],
                 tr=circ.evaluate(Xtr, pool_y), va=circ.evaluate(Xva, val_y),
                 te=circ.evaluate(Xte, test_y))
        snaps.append(s)

    def cd_phase(target: int) -> int:
        done = 0
        while done < target:
            nf = min(args.cd_flips, target - done)
            circ.cd_pass(ytr, rbatch(args.cd_batch), nf)
            done += nf
        flips_total[0] += done
        return done

    def lerp(a: int, b: int, t: float) -> int:
        return int(round(a + (b - a) * t))

    phase = 0
    while True:
        phase += 1
        if args.phases > 0:                                        # phase-based ramp
            t = (phase - 1) / max(1, args.phases - 1)
        else:                                                      # gate-progress ramp
            t = circ.n_gates_built / max(1, args.max_gates)
        built = circ.build_sweep(ytr, rbatch(args.build_batch), lerp(args.build_start, args.build_end, t),
                                 depth_pen=args.depth_penalty, usage_pen=args.usage_penalty,
                                 max_feats=args.max_feats)
        flips = cd_phase(lerp(args.cd_start, args.cd_end, t))
        if phase % args.eval_every == 0:
            show(f"p{phase}", built, flips)
        record(phase)
        if args.phases > 0:
            if phase >= args.phases:
                break
        elif circ.n_gates_built >= args.max_gates:
            break

    # Build budget spent: stop growing, run a long CD-only anneal (watch for a second peak).
    if args.final_cd_flips > 0:
        print(f"final CD annealing: {args.final_cd_flips} flips", flush=True)
        done, k = 0, 0
        while done < args.final_cd_flips:
            nf = min(args.cd_flips, args.final_cd_flips - done)
            circ.cd_pass(ytr, rbatch(args.cd_batch), nf)
            done += nf
            flips_total[0] += nf
            k += 1
            if k % 100 == 0:
                phase += 1
                show(f"c{done // 1000}k", 0, nf)
                record(phase)

    tr, va, te = (circ.evaluate(Xtr, pool_y), circ.evaluate(Xva, val_y),
                  circ.evaluate(Xte, test_y))
    g = circ.def_in[:, 0] >= 0
    gd = circ.depth[g] if g.any() else circ.depth[:1]
    print(f"\nFINAL  gates={circ.n_gates_built}  ops={len(circ.ops)}  "
          f"train={tr:.2f}  val={va:.2f}  test={te:.2f}", flush=True)
    print(f"depth(gates): mean={gd.float().mean():.2f}  max={int(gd.max())}  "
          f"raw-copies-left={int((~g).sum())}  usage-max={int(circ.usage.max())}", flush=True)
    print(f"BEST  val={best['va']:.2f}  test={best['te']:.2f}  "
          f"at {best['tag']}  gates={best['gates']}", flush=True)

    if args.viz and snaps:
        args.viz_out.parent.mkdir(parents=True, exist_ok=True)
        render_animation(snaps, args.viz_out)


if __name__ == "__main__":
    main()
