"""A fixed-topology NAND network whose only free parameters are the WIRES, optimised by a
dead-simple hill-climbing mutation search. No gradients, no gate tables, no special optimizer.

  input  : thermometer bits of the image, (B, 3*num_bits, 32, 32) -> flattened to (B, n_in)
  hidden : D layers of W gates each; every gate is NAND(a, b) = ~(a & b) with two source wires
  output : ONE thin layer of R gates (R << W) grouped mod 10 into class votes, argmax -> (B, 10)

WIRING RULE (keeps the DAG acyclic): a gate in layer l may read ANY earlier signal -- an input bit
OR the output of ANY gate in layers 0..l-1 -- but nothing at its own depth or later. Sources are
strictly lower-depth, so the graph can never contain a cycle. A gate no downstream gate reads is
dead weight; that's fine, the search ignores it.

THE SEARCH (that's the whole algorithm):
  * A GENOME is the full wiring: per layer a (2, width) index tensor (src0, src1) of signal ids.
  * Each generation, make k-1 MUTANTS of the current genome. A mutant picks `mut_count` gates
    UNIFORMLY AT RANDOM over all NAND gates and rewires one endpoint of each to a fresh random
    valid source. The current (unmutated) genome is always the k-th candidate.
  * Score all k on the same minibatch; the best one becomes the current genome. The incumbent is
    always in the race, so fitness never decreases -- pure hill climbing over wirings.

TWO SPEED/MEMORY TRICKS so this scales insanely wide:

  1. BITPACK THE BATCH. 64 samples are packed into one int64 word, so one signal for 64 images is
     ONE int64 and a NAND over 64 samples is a SINGLE bitwise `~(a & b)` -- 64-way SIMD for free.
     acts is (n_signals, ceil(B/64)) int64: ~1 bit per sample per signal.

  2. THIN OUTPUT LAYER. The readout unpacks bits back to per-sample votes, costing ~B*(layer width)
     -- with a wide last layer that dwarfs the whole forward. So votes come from a dedicated NARROW
     output layer of R gates, not from the wide hidden layers.

Plus the DELTA FORWARD: a mutant differs from the incumbent in ONE layer, so only that layer and
those above it are recomputed (the layers below are still the incumbent's activations).

    python scratch_genetic/nand_ga.py --selftest     # packed forward == naive uint8 forward
    python scratch_genetic/nand_ga.py --bench        # time gens/sec + peak memory, no data
    python scratch_genetic/nand_ga.py --width 4096 --depth 8 --out scratch_genetic/runs/g0
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import features  # noqa: E402  (fixed edge/gradient input featurization)
from model import Thermometer  # noqa: E402
from train import load_cifar10  # noqa: E402

CLS = 10
WORD = 64        # samples packed per int64 word along the batch axis
BITS_W1 = 10.0   # fitness: points for a correct target-1 gate (target-0 scores 1); ~CLS balances
                 # the ~CLS x more 0-targets than 1-targets


# ==========================================================================================
# Bitpacking helpers: pack the batch axis 64-to-an-int64, unpack for the readout.
# ==========================================================================================
def pack(Xu: torch.Tensor) -> tuple[torch.Tensor, int]:
    """(B, n) uint8 bits -> (ceil(B/64), n) int64, one signal for 64 samples per word.
    Returns (packed, B). Padding samples (to fill the last word) are 0 and dropped at readout."""
    B, n = Xu.shape
    pad = (-B) % WORD
    if pad:
        Xu = torch.cat([Xu, torch.zeros(pad, n, dtype=torch.uint8, device=Xu.device)], 0)
    Bw = (B + pad) // WORD
    Xu = Xu.view(Bw, WORD, n)
    # Accumulate the 64 sample-bits one shift at a time into an (Bw, n) int64 buffer. Disjoint bits =>
    # OR == shift-then-sum. Avoids the giant (Bw, WORD, n) int64 intermediate that OOMs at large n_in.
    out = torch.zeros(Bw, n, dtype=torch.int64, device=Xu.device)
    for w in range(WORD):
        out |= Xu[:, w, :].to(torch.int64) << w
    return out, B


def unpack_bits(word_col: torch.Tensor) -> torch.Tensor:
    """(Bw, m) int64 -> (Bw*64, m) uint8: expand every word's 64 packed samples along the batch."""
    Bw, m = word_col.shape
    shifts = torch.arange(WORD, device=word_col.device).view(1, WORD, 1)
    return ((word_col.unsqueeze(1) >> shifts) & 1).to(torch.uint8).reshape(Bw * WORD, m)


# ==========================================================================================
# Network dims. Layers = D hidden of width W, then ONE output layer of width R (grouped mod CLS).
# `offs[l]` = number of signals that exist BEFORE layer l = the valid source bound for layer l;
# offs runs [n_in, n_in+W, ..., n_in+D*W, n_in+D*W+R] (len L+1, L = D+1 layers).
# A genome `srcs` is a list of L (2, width_l) int32 tensors of global signal ids.
# ==========================================================================================
def build_dims(n_in: int, W: int, D: int, R: int) -> tuple[list[int], list[int]]:
    widths = [W] * D + [R]
    offs = [n_in]
    for w in widths:
        offs.append(offs[-1] + w)
    return widths, offs


# Wiring source ids are INT32 (half the memory of int64) -- at 1e8 gates the genome is 2*gates ids,
# so int32 vs int64 is 0.8GB vs 1.6GB. Valid because a source id is < n_signals < 2**31 (asserted).
GENE_DTYPE = torch.int32


def init_genome(widths, offs, dev: str, g: torch.Generator) -> list[torch.Tensor]:
    """Random valid wiring: two source ids per gate, drawn from ALL earlier signals."""
    return [torch.randint(0, offs[l], (2, widths[l]), device=dev, generator=g, dtype=GENE_DTYPE)
            for l in range(len(widths))]


# SIGNAL-MAJOR activation buffer: acts is (n_signals, Bw). Gathering a signal is then a CONTIGUOUS
# Bw-wide row (coalesced) instead of a strided column -- the single biggest speed lever (profiled
# 4-9x). On CUDA a fused Triton kernel (gather both wires + NAND + store, one launch) does it; a
# pure-torch signal-major path is the fallback (CPU / no Triton).
try:
    from nand_kernels import forward_acts_sig_triton as _fwd_triton
    _HAVE_TRITON = True
except Exception:                                                    # triton missing -> torch path
    _HAVE_TRITON = False


def _forward_torch(srcs, Xp, offs, lstart, acts):
    if acts is None:
        Bw, n_in = Xp.shape
        acts = torch.empty(offs[-1], Bw, dtype=torch.int64, device=Xp.device)
        acts[:n_in] = Xp.t()
    for l in range(lstart, len(srcs)):
        s = srcs[l]
        a = acts.index_select(0, s[0].long())                        # (W, Bw) coalesced row gather
        b = acts.index_select(0, s[1].long())
        acts[offs[l]: offs[l + 1]] = ~(a & b)
    return acts


@torch.no_grad()
def forward_acts(srcs, Xp: torch.Tensor, offs, lstart: int = 0,
                 acts: torch.Tensor | None = None) -> torch.Tensor:
    """Run the packed NAND net -> the FULL signal buffer acts (n_signals, Bw) int64, signal-major.

    With lstart>0 and a pre-filled `acts`, recompute ONLY layers lstart..end IN PLACE -- the delta
    forward. Sound because a gate in layer l reads only signals < offs[l]: layers below lstart are
    untouched (still the incumbent's), layers lstart.. are overwritten with the new wiring."""
    if _HAVE_TRITON and (Xp.is_cuda if acts is None else acts.is_cuda):
        return _fwd_triton(srcs, Xp, offs, lstart, acts)
    return _forward_torch(srcs, Xp, offs, lstart, acts)


def out_bits(acts: torch.Tensor, offs) -> torch.Tensor:
    # output layer is (R, Bw) signal-major; transpose to (Bw, R) so votes_of unpacks per-sample
    return acts[offs[-2]: offs[-1]].t().contiguous()


@torch.no_grad()
def votes_of(ob: torch.Tensor, B: int) -> torch.Tensor:
    """Class votes (B, CLS) from output bits: unpack ONLY the thin output layer, then sum each
    class's gates in one reshape. Output gate g votes for class g%CLS => (B,R)->(B,R/CLS,CLS)."""
    out = unpack_bits(ob)[:B]                                        # (B, R) uint8
    return out.view(B, out.shape[1] // CLS, CLS).sum(1).to(torch.int32)


# ==========================================================================================
# FITNESS = integer BIT-MATCH point count. The target is the NATURAL class code: output gate r
# should be 1 iff the sample's class == r % CLS (the same grouping votes uses). A candidate scores
#     BITS_W1 points per gate that is 1 where target is 1  (correctly firing the class's gates)
#     1 point         per gate that is 0 where target is 0  (correctly off elsewhere)
# Pure packed AND + popcount -- NO unpack, NO argmax -- so it's fast and gives every gate a target
# every step (dense signal, not just the argmax class). Prediction is still argmax votes, so
# accuracy() at eval is unchanged.
# ==========================================================================================
_POP_LUT = None                                                      # 256-entry byte popcount LUT


def _popcount_sum(x: torch.Tensor) -> torch.Tensor:
    """Total set bits over all elements of an int64 tensor (via a 256-entry byte LUT -- robust,
    no sign-extension issues from right shifts)."""
    global _POP_LUT
    if _POP_LUT is None or _POP_LUT.device != x.device:
        _POP_LUT = torch.tensor([bin(i).count("1") for i in range(256)],
                                dtype=torch.int64, device=x.device)
    return _POP_LUT[x.contiguous().view(torch.uint8).long()].sum()


def _target_packed(y: torch.Tensor, Bw: int, R: int) -> torch.Tensor:
    """(R, Bw) int64 signal-major target: gate r is 1 iff sample's class == r % CLS. Build the small
    (CLS, Bw) class masks, then index to R gates -- cheap, no (R,B) intermediate."""
    oh = (y.view(1, -1) == torch.arange(CLS, device=y.device).view(CLS, 1))   # (CLS, B) one-hot
    shifts = torch.arange(WORD, device=y.device).view(1, 1, WORD)
    classmask = (oh.view(CLS, Bw, WORD).to(torch.int64) << shifts).sum(2)     # (CLS, Bw) packed
    return classmask[torch.arange(R, device=y.device) % CLS]                  # (R, Bw)


@torch.no_grad()
def score_cand(acts, offs, target) -> float:
    """Bit-match FITNESS of one candidate, read straight off the packed output layer (R, Bw)."""
    os_sig = acts[offs[-2]: offs[-1]]                               # (R, Bw) signal-major view
    both1 = _popcount_sum(os_sig & target).float()                  # correctly-fired (target 1) bits
    both0 = _popcount_sum(~os_sig & ~target).float()                # correctly-off (target 0) bits
    n = float(os_sig.shape[0] * os_sig.shape[1] * WORD)
    return float((BITS_W1 * both1 + both0) / n)                     # up-weight getting the 1s right


@torch.no_grad()
def votes(srcs, Xp, B: int, offs) -> torch.Tensor:                  # full-forward convenience
    return votes_of(out_bits(forward_acts(srcs, Xp, offs), offs), B)


EVAL_ELEMS = 50_000_000  # cap acts elements per eval chunk (~0.4GB int64) so eval never OOMs


@torch.no_grad()
def accuracy(srcs, Xp, y, B, offs) -> float:
    """Fraction correct over a packed eval set, CHUNKED over words so peak memory is bounded
    (~EVAL_ELEMS) no matter how wide the net -- the eval set is far larger than a fitness
    minibatch, so forwarding it whole would OOM at large width."""
    chunk = max(1, EVAL_ELEMS // offs[-1])                           # words per eval forward
    correct = 0
    for i in range(0, Xp.shape[0], chunk):
        xc = Xp[i:i + chunk]
        ob = out_bits(forward_acts(srcs, xc, offs), offs)            # (chunk, R) packed
        rows = xc.shape[0] * WORD
        pred = votes_of(ob, rows).argmax(1)                          # argmax votes = nearest class code
        yc = y[i * WORD: i * WORD + rows]                            # trailing padding has no label
        correct += int((pred[:yc.shape[0]] == yc).sum())
    return correct / B


# ==========================================================================================
# MUTATION. To scale insanely wide we DON'T clone the genome per candidate (that would hold k
# copies at once). A mutant is applied IN PLACE to the single resident genome as a tiny sparse
# patch, recording the old values so it can be undone in O(#changed) after scoring. Peak memory is
# ONE genome + one small patch, independent of k.
#
# Gates are chosen UNIFORMLY over all NAND gates. A mutant touches one layer (so the delta forward
# can skip the layers below it), and that layer is drawn with probability proportional to its WIDTH
# -- which is exactly uniform sampling over gates.
# ==========================================================================================
def mutate_one(s: torch.Tensor, bound: int, m: int, dev: str, g: torch.Generator,
               not_prob: float = 0.0):
    """Rewire an ABSOLUTE COUNT m of ONE layer's endpoints to fresh random valid sources; return a
    patch entry or None. Absolute (not a fraction): a fraction of a 16K-wide layer is hundreds of
    simultaneous rewires that randomize the child -- small steps (m=1) keep a mutant a useful
    neighbour of its parent.

    not_prob: with this probability a rewired endpoint is set to the SIBLING wire's current source
    instead of a random one. Since NAND(s,s) = NOT s, that collapses the gate to a NOT -- otherwise
    unreachable, because two random wires coincide with probability ~1/pool."""
    m = min(m, s.shape[1])
    if m <= 0:
        return None
    idx = torch.randint(0, s.shape[1], (m,), device=dev, generator=g)               # m gate slots
    endpoint = (torch.rand(m, device=dev, generator=g) < 0.5).long()                # wire 0 or 1
    old = s[endpoint, idx].clone()
    new = torch.randint(0, bound, (m,), device=dev, generator=g, dtype=GENE_DTYPE)
    if not_prob > 0.0:                                                              # -> NOT gate
        sibling = s[1 - endpoint, idx]                                              # other wire's src
        new = torch.where(torch.rand(m, device=dev, generator=g) < not_prob, sibling, new)
    s[endpoint, idx] = new
    return (s, endpoint, idx, old, new)


def best_of_k(current, Xb, yb, offs, k, m, layers, dev, g, low_mem=False,
              not_prob=0.0) -> float:
    """One hill-climbing step; returns the fitness after this generation. Compute the incumbent's
    activations ONCE; each of the k-1 mutants rewires m endpoints of a SINGLE layer l (from
    `layers`, pre-drawn uniformly over gates) and recomputes only layers l..end IN PLACE (the delta
    forward), scores by bit-match fitness, then restores the tail for the next mutant. Commit the
    best mutant if it is at least as fit as the incumbent -- a TIE is accepted (NEUTRAL DRIFT: the
    wiring wanders across equal-fitness genomes, the stepping stones evolution uses to cross
    plateaus). Never commits a worse genome, so fitness is non-decreasing.

    low_mem: restore acts by RE-forwarding the tail (undo the genome first) instead of cloning it --
    one extra tail-forward per mutant, but ~half the activation memory (what fits 1e8 gates in 24GB).
    """
    acts = forward_acts(current, Xb, offs)                          # incumbent, once
    target = _target_packed(yb, acts.shape[1], offs[-1] - offs[-2])
    inc_fit = score_cand(acts, offs, target)                       # fixed reference for this gen
    best_mut, winner = float("-inf"), None
    for l in layers.tolist():                                      # k-1 pre-drawn mutation layers
        entry = mutate_one(current[l], offs[l], m, dev, g, not_prob)
        if entry is None:
            continue
        backup = None if low_mem else acts[offs[l]:].clone()       # save tail (unless low-mem)
        forward_acts(current, Xb, offs, lstart=l, acts=acts)       # recompute only the tail
        fit = score_cand(acts, offs, target)
        current[l][entry[1], entry[2]] = entry[3]                  # undo genome (back to incumbent)
        if low_mem:
            forward_acts(current, Xb, offs, lstart=l, acts=acts)   # restore acts by recompute
        else:
            acts[offs[l]:] = backup                                # restore incumbent tail from clone
        if fit > best_mut:
            best_mut, winner = fit, entry
    if winner is not None and best_mut >= inc_fit:                 # >= : accept neutral drift
        winner[0][winner[1], winner[2]] = winner[4]                # commit
        return best_mut
    return inc_fit


# ==========================================================================================
# Memory: acts is the dominant cost (n_signals * Bw * 8 bytes), and best_of_k holds a second copy
# of the tail unless --low-mem. Pick the largest fitness batch that fits, so a 1e8-gate net still
# runs on a 24GB card (at a small batch) instead of OOMing.
# ==========================================================================================
def plan_memory(offs, widths, want_batch: int, dev: str, force_low_mem: bool) -> tuple[int, bool]:
    """Pick (batch, low_mem) so the run FITS. acts is offs[-1] * Bw * 8 bytes, and best_of_k holds a
    second copy of the tail unless low_mem -- so the default path costs ~2x acts. Preference order:

      1. the full requested batch, cloning the tail        (fastest)
      2. the full requested batch with low_mem             (2x the forwards, half the memory)
      3. the biggest batch that fits with low_mem          (what a 1e8-gate net gets)

    A big batch is worth more than a cheap generation: fitness on a tiny batch is noise, and the
    search commits to it."""
    if not dev.startswith("cuda"):
        return want_batch, force_low_mem
    free = torch.cuda.mem_get_info()[0]
    genome = sum(2 * w for w in widths) * 4                          # int32 wiring, resident
    budget = 0.9 * (free - genome - 2.5e9)                           # reserve: eval + workspace
    want_bw = max(1, want_batch // WORD)
    for low_mem in ([True] if force_low_mem else [False, True]):
        per_word = offs[-1] * 8 * (1 if low_mem else 2)              # acts (+ tail backup clone)
        bw = int(budget // per_word)
        if bw >= want_bw:
            return want_bw * WORD, low_mem
    bw = max(1, int(budget // (offs[-1] * 8)))                       # biggest that fits, low_mem
    return bw * WORD, True


# ==========================================================================================
# Self-test: the packed forward must agree bit-for-bit with a naive per-sample uint8 forward.
# ==========================================================================================
def selftest(dev: str) -> None:
    g = torch.Generator(device=dev).manual_seed(1)
    n_in, W, D, R, B = 50, 32, 4, 30, 100
    widths, offs = build_dims(n_in, W, D, R)
    Xu = (torch.rand(B, n_in, device=dev, generator=g) < 0.5).to(torch.uint8)
    srcs = init_genome(widths, offs, dev, g)

    # naive uint8 reference: full forward, then group the output layer's bits mod CLS
    acts = torch.empty(B, offs[-1], dtype=torch.uint8, device=dev)
    acts[:, :n_in] = Xu
    for l, s in enumerate(srcs):
        a = acts.index_select(1, s[0])
        b = acts.index_select(1, s[1])
        acts[:, offs[l]: offs[l + 1]] = 1 - a * b
    out_ref = acts[:, offs[-2]: offs[-1]]
    v_ref = out_ref.view(B, R // CLS, CLS).sum(1).to(torch.int32)

    Xp, Bret = pack(Xu)
    v_pack = votes(srcs, Xp, Bret, offs)
    assert Bret == B and torch.equal(v_ref, v_pack), "packed forward disagrees with naive!"

    # delta forward: recomputing only layers l..end in place must match a fresh full forward of the
    # mutated genome (exercise every split layer l).
    base = forward_acts(srcs, Xp, offs)
    for l in range(len(srcs)):
        e = mutate_one(srcs[l], offs[l], 5, dev, g)                # force a real mutation
        full = forward_acts(srcs, Xp, offs)                        # ground truth
        delta = forward_acts(srcs, Xp, offs, lstart=l, acts=base.clone())
        assert torch.equal(full, delta), f"delta forward disagrees at split layer {l}!"
        srcs[l][e[1], e[2]] = e[3]                                 # undo
    print(f"selftest OK on {dev}: packed == naive AND delta == full at every split layer "
          f"(B={B}, W={W}, D={D}, R={R}); votes max |diff| = {int((v_ref - v_pack).abs().max())}")


# ==========================================================================================
# Benchmark: time gens/sec + peak memory on random data -- no CIFAR needed.
# ==========================================================================================
def bench(args, dev: str) -> None:
    g = torch.Generator(device=dev).manual_seed(0)
    n_in = 3 * args.num_bits * 32 * 32
    widths, offs = build_dims(n_in, args.width, args.depth, args.out_width)
    batch, low_mem = (plan_memory(offs, widths, args.batch, dev, args.low_mem) if args.auto_batch
                      else (args.batch, args.low_mem))
    bw = max(1, batch // WORD)
    Xp = torch.randint(-(2**63), 2**63 - 1, (bw, n_in), dtype=torch.int64, device=dev)
    y = torch.randint(0, CLS, (bw * WORD,), device=dev)
    cur = init_genome(widths, offs, dev, g)
    lw = torch.tensor([float(w) for w in widths], device=dev)      # uniform over GATES
    gates = args.width * args.depth + args.out_width
    print(f"bench W={args.width} D={args.depth} R={args.out_width} gates={gates:,} k={args.k} "
          f"batch={bw * WORD} device={dev} | genome ~{sum(2*w for w in widths)*4/1e9:.2f}GB "
          f"acts ~{bw*offs[-1]*8/1e9:.2f}GB low_mem={low_mem}", flush=True)

    def one_gen():
        layers = torch.multinomial(lw, args.k - 1, replacement=True, generator=g)
        best_of_k(cur, Xp, y, offs, args.k, args.mut_count, layers, dev, g,
                  low_mem=low_mem, not_prob=args.not_prob)

    for _ in range(3):                                             # warmup (CUDA autotune/alloc)
        one_gen()
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(args.bench_gens):
        one_gen()
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
    dt = 1000 * (time.time() - t0) / args.bench_gens
    peak = torch.cuda.max_memory_allocated() / 1e9 if dev.startswith("cuda") else 0.0
    print(f"{dt:7.2f} ms/gen ({1000/dt:6.1f} gen/s)" + (f" | peak {peak:.2f}GB" if peak else ""),
          flush=True)


# ==========================================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/cifar-10-batches-py"))
    p.add_argument("--download", action="store_true")
    p.add_argument("--selftest", action="store_true", help="verify packed==naive and exit")
    p.add_argument("--bench", action="store_true", help="time gens/sec on random data and exit")
    p.add_argument("--bench-gens", type=int, default=200)
    p.add_argument("--num-bits", type=int, default=4, help="thermometer bits per channel")
    p.add_argument("--features", type=str, default="",
                   help="comma list of fixed input feature families (e.g. raw,edge); default raw")
    # --- net shape ---
    p.add_argument("--width", type=int, default=4096, help="gates per hidden layer (W)")
    p.add_argument("--depth", type=int, default=8, help="number of hidden NAND layers (D)")
    p.add_argument("--out-width", type=int, default=2560,
                   help="gates in the thin output layer (R, multiple of 10)")
    # --- search ---
    p.add_argument("--k", type=int, default=16, help="candidates per generation (incl. incumbent)")
    p.add_argument("--mut-count", type=int, default=1,
                   help="endpoints rewired per mutant (absolute count, not a fraction)")
    p.add_argument("--not-prob", type=float, default=0.5,
                   help="chance a rewire points at the sibling wire => NAND(s,s) = NOT s")
    p.add_argument("--batch", type=int, default=8192, help="minibatch size for fitness scoring")
    p.add_argument("--no-auto-batch", dest="auto_batch", action="store_false",
                   help="don't shrink --batch to fit GPU memory")
    p.add_argument("--low-mem", action="store_true",
                   help="restore acts by recompute instead of a clone (~half the memory)")
    # --- stopping: run to convergence, with a hard wall-clock cap ---
    p.add_argument("--gens", type=int, default=100_000_000, help="generation cap (rarely binding)")
    p.add_argument("--patience", type=int, default=20,
                   help="stop after this many evals with no val improvement (0 = never)")
    p.add_argument("--max-minutes", type=float, default=0.0, help="hard wall-clock cap (0 = none)")
    # --- io ---
    p.add_argument("--eval-every", type=int, default=200, help="gens between full val/test eval")
    p.add_argument("--ckpt-every", type=int, default=1000, help="gens between genome checkpoints")
    p.add_argument("--out", type=Path, help="prefix for .jsonl and .pkl (required to train)")
    p.add_argument("--fresh", action="store_true", help="ignore any saved .pkl and start over")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    dev = args.device
    W, D, R = args.width, args.depth, args.out_width
    assert R % CLS == 0, "--out-width must be a multiple of 10 (one vote group per class)"

    if args.selftest:
        selftest(dev)
        return
    if args.bench:
        bench(args, dev)
        return
    assert args.out is not None, "--out is required to train"

    # ---- data + thermometer bits, packed once (batch axis -> int64 words) -----------------
    tx, ty, ex, ey = load_cifar10(args.data_dir, args.download)
    nv = 5000
    fams = tuple(f for f in args.features.split(",") if f) if args.features else ("raw",)

    @torch.no_grad()
    def feat(images: torch.Tensor) -> torch.Tensor:
        return features.expand(images.to(dev), fams) if fams != ("raw",) else images.to(dev)

    enc = Thermometer(num_bits=args.num_bits).fit(feat(tx[:2000]).cpu()).to(dev)

    @torch.no_grad()
    def encode(images: torch.Tensor) -> torch.Tensor:        # (N, n_in) uint8, chunked for memory
        outs = [enc(feat(images[i:i + 4096])).flatten(1).to(torch.uint8)
                for i in range(0, len(images), 4096)]
        return torch.cat(outs, 0)

    Xtr_u = encode(tx[:-nv])
    Ntr = (Xtr_u.shape[0] // WORD) * WORD                    # whole words only (drops <64 samples)
    Xtr_p, _ = pack(Xtr_u[:Ntr])                             # (Nw, n_in) int64
    ytr = ty[:-nv][:Ntr].to(dev)
    Nw, n_in = Xtr_p.shape
    Xva_p, Bva = pack(encode(tx[-nv:]))
    yva = ty[-nv:].to(dev)
    Xte_p, Bte = pack(encode(ex))
    yte = ey.to(dev)
    ntr_ev = min(Nw, 160)                                    # train-acc on a subset (fast eval)
    Xtr_ev, Btr_ev = Xtr_p[:ntr_ev], ntr_ev * WORD
    ytr_ev = ytr[:ntr_ev * WORD]

    widths, offs = build_dims(n_in, W, D, R)
    assert offs[-1] < 2**31, f"n_signals {offs[-1]} exceeds int32 source-id range (2^31)"
    L = len(widths)
    gates = W * D + R
    igen = torch.Generator(device=dev).manual_seed(args.seed)          # genome init
    mgen = torch.Generator(device=dev).manual_seed(args.seed)          # mutations + layer choice
    dgen = torch.Generator(device=dev).manual_seed(args.seed + 1)      # minibatch selection

    batch, low_mem = (plan_memory(offs, widths, args.batch, dev, args.low_mem) if args.auto_batch
                      else (args.batch, args.low_mem))
    bw = max(1, batch // WORD)                                         # words per fitness minibatch
    # A mutant's layer is drawn with probability proportional to its width == picking a NAND gate
    # UNIFORMLY AT RANDOM over the whole net (each mutant touches one layer so the delta forward
    # can skip everything below it).
    lweights = torch.tensor([float(w) for w in widths], device=dev)

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    # append suffixes as STRINGS, not Path.with_suffix -- an --out like "g1e6_d4" is fine but one
    # with a dot would have it stripped, silently collapsing distinct runs onto one file.
    ckpt = Path(str(out) + ".pkl")
    jsonl = Path(str(out) + ".jsonl")

    def save(current, gen, best_val, best_test, stop=""):     # genome checkpoint (always on)
        with open(ckpt, "wb") as f:
            pickle.dump({"args": vars(args) | {"out": str(out)}, "gen": gen, "stop": stop,
                         "best_val": best_val, "best_test": best_test,
                         "srcs": [s.cpu() for s in current], "thr": enc.thresholds.cpu(),
                         "num_bits": args.num_bits, "n_in": n_in, "offs": offs,
                         "W": W, "D": D, "R": R, "gates": gates}, f)

    # ---- resume-by-default: pick up the saved genome + jsonl if present -------------------
    if not args.fresh and ckpt.exists():
        c = pickle.load(open(ckpt, "rb"))
        # A CONVERGED run is done -- resubmitting the array must not restart it. A run stopped by
        # the TIME CAP is not done: resume and keep searching (that's how you give it more budget).
        if c.get("stop", "").startswith("converged"):
            print(f"already {c['stop']} at gen {c['gen']}, "
                  f"best_val {100*c['best_val']:.2f} -- nothing to do", flush=True)
            return
        current = [s.to(dev, GENE_DTYPE) for s in c["srcs"]]
        start_gen, best_val, best_test = c["gen"] + 1, c["best_val"], c["best_test"]
        print(f"resume from {ckpt} at gen {start_gen} (best_val {100*best_val:.2f})", flush=True)
    else:
        current = init_genome(widths, offs, dev, igen)
        start_gen, best_val, best_test = 0, 0.0, 0.0
        jsonl.write_text("")

    print(f"nand_ga n_in={n_in} W={W} D={D} R={R} gates={gates:,} k={args.k} "
          f"mut_count={args.mut_count} not={args.not_prob} batch={bw * WORD} device={dev} "
          f"| acts ~{bw * offs[-1] * 8 / 1e9:.2f}GB genome ~{sum(2*w for w in widths)*4/1e9:.2f}GB "
          f"low_mem={low_mem} patience={args.patience} max_min={args.max_minutes}", flush=True)

    # ---- hill-climbing search over wirings ------------------------------------------------
    t0, stale, stop = time.time(), 0, ""
    for gen in range(start_gen, args.gens):
        wsel = torch.randint(0, Nw, (bw,), device=dev, generator=dgen)   # word-blocks of the batch
        Xb = Xtr_p[wsel]
        yb = ytr[(wsel[:, None] * WORD + torch.arange(WORD, device=dev)).reshape(-1)]
        layers = torch.multinomial(lweights, args.k - 1, replacement=True, generator=mgen)
        fit = best_of_k(current, Xb, yb, offs, args.k, args.mut_count, layers, dev, mgen,
                        low_mem=low_mem, not_prob=args.not_prob)

        if args.ckpt_every and gen % args.ckpt_every == 0:
            save(current, gen, best_val, best_test)

        over_time = args.max_minutes > 0 and (time.time() - t0) / 60 >= args.max_minutes
        if gen % args.eval_every == 0 or gen == args.gens - 1 or over_time:
            tr = accuracy(current, Xtr_ev, ytr_ev, Btr_ev, offs)
            va = accuracy(current, Xva_p, yva, Bva, offs)
            te = accuracy(current, Xte_p, yte, Bte, offs)
            if va > best_val:                                    # EARLY STOPPING on val accuracy
                best_val, best_test, stale = va, te, 0
            else:
                stale += 1
            rec = {"gen": gen, "gates": gates, "W": W, "D": D, "R": R, "m": args.mut_count,
                   "batch": bw * WORD,
                   "fitness": round(fit, 4), "train": round(100 * tr, 2), "val": round(100 * va, 2),
                   "test": round(100 * te, 2), "best_val": round(100 * best_val, 2),
                   "best_test": round(100 * best_test, 2), "stale": stale,
                   "min": round((time.time() - t0) / 60, 2),
                   "gps": round((gen - start_gen + 1) / max(time.time() - t0, 1e-6), 2)}
            with open(jsonl, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"gen {gen:7d} | fit {rec['fitness']:6.4f} | train {rec['train']:5.2f} "
                  f"val {rec['val']:5.2f} test {rec['test']:5.2f} (best val {rec['best_val']:5.2f}) "
                  f"| stale {stale:2d} | {rec['gps']:6.2f} gen/s | {rec['min']:6.1f}m", flush=True)
            save(current, gen, best_val, best_test)

            if args.patience and stale >= args.patience:
                stop = f"converged (no val gain in {args.patience} evals)"
            elif over_time:
                stop = f"time cap ({args.max_minutes} min)"
            if stop:
                break

    stop = stop or "gen cap"
    save(current, gen, best_val, best_test, stop=stop)
    print(f"STOP: {stop} | gates={gates:,} W={W} D={D} | best_val={100*best_val:.2f} "
          f"best_test={100*best_test:.2f} | {(time.time()-t0)/60:.1f}m -> {ckpt}", flush=True)


if __name__ == "__main__":
    main()
