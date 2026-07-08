"""scratch_genetic: a fixed-topology NAND network whose only free parameters are the WIRES,
optimised by a dead-simple GENETIC ALGORITHM (no gradients, no gate-table learning).

The whole model is a stack of 2-input NAND gates -- the universal Boolean gate, so the network
can in principle represent any function; all the learning is in WHICH signals feed each gate.

  input  : thermometer bits of the image, (B, 3*num_bits, 32, 32) -> flattened to (B, n_in)
  hidden : D layers of W gates each; every gate is NAND(a, b) = ~(a & b) with two source wires
  output : ONE thin layer of R gates (R << W) grouped mod 10 into class votes, argmax -> (B, 10)

WIRING RULE (keeps the DAG acyclic): a gate in layer l may read ANY earlier signal -- an input
bit OR the output of ANY gate in layers 0..l-1 -- but nothing at its own depth or later. Because
sources are strictly lower-depth the graph can never contain a cycle. A gate whose output no
downstream gate happens to read is simply dead weight; that's fine, the search prunes/ignores it.

TWO SPEED/MEMORY TRICKS so this scales insanely wide:

  1. BITPACK THE BATCH. 64 samples are packed into one int64 word, so one signal for 64 images is
     ONE int64 and a NAND over 64 samples is a SINGLE bitwise `~(a & b)` -- 64-way SIMD for free,
     no per-sample loop, no float mults. acts is (ceil(B/64), n_signals) int64; the forward is D
     bitwise ops on it. Memory is ~B/64 * n_signals * 8 bytes -- 1 bit per sample per signal.

  2. THIN OUTPUT LAYER. The readout has to UNPACK bits back to per-sample votes, which costs ~B*
     (layer width) -- with a wide last layer that dwarfs the whole forward (~64/D x more work).
     So the votes come from a dedicated NARROW output layer of R gates (R = out_width, e.g. 2560),
     not from the wide hidden layers. Readout then unpacks only B*R bits and reshapes them into
     (B, R/10, 10) to sum each class's gates in one shot -- no giant float index_add.

GENETIC ALGORITHM -- a (1+lambda) evolutionary strategy with ELITISM:
  * A GENOME is the full wiring: per layer a (2, width) index tensor (src0, src1) of signal ids.
  * Each generation we take the CURRENT genome and make k-1 MUTANTS (each rewires a small random
    fraction of endpoints to fresh valid sources), giving k candidates. THE CURRENT (unmutated)
    GENOME IS ALWAYS ONE OF THE k CANDIDATES -- "one of the candidates is the already-used gate."
  * All k are scored on the same minibatch (accuracy of the popcount readout); the best one wins
    and becomes the current genome. Because the incumbent is always in the race, a generation can
    never make things worse on its own batch -- pure hill-climbing over wirings.

Self-contained (torch + repo encoder/data), device-agnostic (CPU here, CUDA when present).

    python scratch_genetic/nand_ga.py --selftest          # packed forward == naive uint8 forward
    python scratch_genetic/nand_ga.py --bench             # time gens/sec + peak memory, no data
    python scratch_genetic/nand_ga.py --out scratch_genetic/runs/g0            # full 64K x 5 run
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Thermometer  # noqa: E402
from train import load_cifar10  # noqa: E402

CLS = 10
WORD = 64  # samples packed per int64 word along the batch axis
_WORLD = 1  # data-parallel world size; >1 => GA fitness is all-reduced across the batch shards
_FITNESS = "margin"  # selection fitness: "margin", "ce", or "bits" (weighted packed bit-match)
_BITS_W1 = 10.0      # 'bits': points for a correct target-1 gate (target-0 = 1); default ~CLS


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
    shifts = torch.arange(WORD, device=Xu.device).view(1, WORD, 1)
    # disjoint bits => shift-then-sum equals bitwise OR (two's complement, bit 63 = sign, still ok)
    return (Xu.view(Bw, WORD, n).to(torch.int64) << shifts).sum(1), B


def unpack_bits(word_col: torch.Tensor) -> torch.Tensor:
    """(Bw, m) int64 -> (Bw*64, m) uint8: expand every word's 64 packed samples along the batch."""
    Bw, m = word_col.shape
    shifts = torch.arange(WORD, device=word_col.device).view(1, WORD, 1)
    return ((word_col.unsqueeze(1) >> shifts) & 1).to(torch.uint8).reshape(Bw * WORD, m)


# ==========================================================================================
# Network dims. Layers = D hidden of width W, then ONE output layer of width R (grouped mod CLS).
# `offs[l]` = number of signals that exist BEFORE layer l = the valid source bound for layer l;
# offs runs [n_in, n_in+W, ..., n_in+D*W, n_in+D*W+R] (len L+1, L = D+1 layers).
# A genome `srcs` is a list of L (2, width_l) long tensors of global signal ids.
# ==========================================================================================
def build_dims(n_in: int, W: int, D: int, R: int) -> tuple[list[int], list[int]]:
    widths = [W] * D + [R]
    offs = [n_in]
    for w in widths:
        offs.append(offs[-1] + w)
    return widths, offs


# Wiring source ids are stored as INT32 (half the memory of int64) -- at 1e9 gates the genome is
# 2*D*W ids, so int32 vs int64 is 8GB vs 16GB, the difference between fitting a 24GB card or not.
# Valid because a source id is < n_signals; we assert n_signals < 2**31 at startup.
GENE_DTYPE = torch.int32

# DEPTH-LOCALITY: with _LOCALITY = K, a gate in layer l may only wire to the previous K layers'
# signals (range [offs[l-K], offs[l])), no width-index constraint. K=0 = unconstrained (any earlier
# signal). Locality forces signals to propagate THROUGH the depth instead of long input->output
# jumps that make a deep net behave shallow -- the biggest lever for a weak mutation search.
_LOCALITY = 0


def lo_of(offs, l: int) -> int:
    """Lowest signal id a gate in layer l may read: offs[l-K] under K-locality, else 0 (inputs)."""
    return offs[l - _LOCALITY] if _LOCALITY and l >= _LOCALITY else 0


def init_genome(widths, offs, dev: str, g: torch.Generator) -> list[torch.Tensor]:
    """Random valid wiring: two source ids per gate from the allowed (locality-limited) signal range."""
    return [torch.randint(lo_of(offs, l), offs[l], (2, widths[l]), device=dev, generator=g,
                          dtype=GENE_DTYPE) for l in range(len(widths))]


# SIGNAL-MAJOR activation buffer: acts is (n_signals, Bw). Gathering a signal is then a CONTIGUOUS
# Bw-wide row (coalesced) instead of a strided column -- the single biggest speed lever (profiled
# 4-9x). On CUDA a fused Triton kernel (gather both wires + NAND + store, one launch, no 2W
# intermediate) does it; a pure-torch signal-major path is the fallback (CPU / no Triton).
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


def acc_of(ob: torch.Tensor, y, B: int) -> float:
    return float((votes_of(ob, B).argmax(1) == y).float().mean())


# ==========================================================================================
# BIT-MATCH fitness (--fitness bits): the target is the NATURAL class code -- output gate r should
# be 1 iff the sample's class == r % CLS (the same grouping votes uses). Score each candidate by an
# integer point count over the packed output bits:
#     w1 points per gate that is 1 where target is 1  (correctly firing the class's gates)
#      1 point  per gate that is 0 where target is 0  (correctly off elsewhere)
# and keep the candidate with the most points. Pure packed AND + popcount (no unpack, no argmax) ->
# fast and every gate gets a target every step (dense signal). w1 defaults to CLS: there are ~CLS x
# more 0-targets than 1-targets, so weighting a 1-match by CLS balances the 0/1 pressure. Prediction
# is still argmax votes (== nearest natural codeword), so accuracy() is unchanged.
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
    oh = (y.view(1, -1) == torch.arange(CLS, device=y.device).view(CLS, 1))  # (CLS, B) one-hot
    shifts = torch.arange(WORD, device=y.device).view(1, 1, WORD)
    classmask = (oh.view(CLS, Bw, WORD).to(torch.int64) << shifts).sum(2)     # (CLS, Bw) packed
    return classmask[torch.arange(R, device=y.device) % CLS]                  # (R, Bw)


@torch.no_grad()
def score_cand(acts, offs, yb, B, target) -> tuple[float, float]:
    """(fitness, acc). 'bits': integer point count over packed output bits (w1 per correct-1, 1 per
    correct-0), no unpack/argmax; acc placeholder=fitness (real acc from accuracy() at eval). Else
    delegate to score_of. DP-safe (all-reduce the sums)."""
    if _FITNESS == "bits":
        os_sig = acts[offs[-2]: offs[-1]]                            # (R, Bw) signal-major view
        both1 = _popcount_sum(os_sig & target).float()              # correctly-fired (target 1) bits
        both0 = _popcount_sum(~os_sig & ~target).float()            # correctly-off (target 0) bits
        n = torch.tensor(float(B * os_sig.shape[0]), device=acts.device)
        if _WORLD > 1:
            s = torch.stack([both1, both0, n]); dist.all_reduce(s); both1, both0, n = s
        f = float((_BITS_W1 * both1 + both0) / n)                    # w1 up-weights getting 1s right
        return f, f
    return score_of(out_bits(acts, offs), yb, B)


@torch.no_grad()
def score_of(ob: torch.Tensor, y, B: int) -> tuple[float, float]:
    """Return (fitness, accuracy). SELECTION uses the smooth fitness = mean vote MARGIN
    (correct-class votes − best wrong-class votes), averaged over the batch. Unlike 0/1 accuracy,
    a single rewire shifts many samples' vote counts, so the batch-mean margin almost always moves
    -- no quantized 'everything ties' plateau. Accuracy is returned only for logging.

    'ce' fitness: -cross_entropy(softmax(votes / sqrt(group_size)), y). The class vote is a sum of
    group_size gate bits, so its spread scales like sqrt(group_size); dividing by that keeps the
    softmax logits O(1) (not saturated) regardless of gates-per-class. Uses the FULL class
    distribution (penalises mass on every wrong class), where margin only sees the top wrong class.

    DATA PARALLEL: with _WORLD>1 each rank passes its own batch shard; we all-reduce the fitness sum,
    correct count, and sample count so every rank sees the SAME global fitness over the full effective
    batch (world*B) and therefore makes the SAME selection -- genomes stay identical without ever
    communicating the wiring (only 3 scalars per call)."""
    v = votes_of(ob, B).float()                                     # (B, CLS) vote counts
    ar = torch.arange(B, device=v.device)
    if _FITNESS == "ce":
        logits = v * (CLS / ob.shape[1]) ** 0.5                     # / sqrt(group_size), group=R/CLS
        fit = logits[ar, y] - torch.logsumexp(logits, 1)           # log softmax of correct = -CE
        fsum = fit.sum()
        ncorrect = (logits.argmax(1) == y).float().sum()
    else:                                                          # margin
        correct = v[ar, y]
        v = v.index_put((ar, y), torch.full_like(correct, float("-inf")))  # mask the true class
        other = v.max(1).values                                    # best wrong-class votes
        fsum = (correct - other).sum()
        ncorrect = (correct > other).float().sum()                 # argmax==y  <=>  margin>0
    if _WORLD > 1:
        stat = torch.stack([fsum, ncorrect, torch.tensor(float(B), device=v.device)])
        dist.all_reduce(stat)                                       # SUM across ranks (default op)
        fsum, ncorrect, tot = stat
        return float(fsum / tot), float(ncorrect / tot)
    return float(fsum / B), float(ncorrect / B)


@torch.no_grad()
def votes(srcs, Xp, B: int, offs) -> torch.Tensor:                    # full-forward convenience
    return votes_of(out_bits(forward_acts(srcs, Xp, offs), offs), B)


EVAL_ELEMS = 200_000_000  # cap acts elements per eval chunk (~1.6GB int64) so eval never OOMs


@torch.no_grad()
def accuracy(srcs, Xp, y, B, offs) -> float:
    """Fraction correct over a packed eval set, CHUNKED over words so peak memory is bounded
    (~EVAL_ELEMS) no matter how wide the net -- the eval set can be far larger than a fitness
    minibatch, so forwarding it whole would OOM at large width."""
    chunk = max(1, EVAL_ELEMS // offs[-1])                           # words per eval forward
    correct = 0
    for i in range(0, Xp.shape[0], chunk):
        xc = Xp[i:i + chunk]
        ob = out_bits(forward_acts(srcs, xc, offs), offs)           # (chunk, R) packed
        rows = xc.shape[0] * WORD
        pred = votes_of(ob, rows).argmax(1)                         # argmax votes = nearest class code
        yc = y[i * WORD: i * WORD + rows]                           # trailing padding has no label
        correct += int((pred[:yc.shape[0]] == yc).sum())
    return correct / B


# ==========================================================================================
# Mutation. To scale insanely wide we DON'T clone the genome per candidate (that would hold k
# copies = k * D*W * 8 bytes at once). Instead a mutant is applied IN PLACE to the single resident
# genome as a tiny sparse patch: rewire a random `rate` fraction of endpoints to fresh valid
# sources, recording the old values so it can be undone in O(#changed) after scoring. Peak memory
# is thus ONE genome + one small patch, independent of k. Only one wire per chosen gate is touched.
# ==========================================================================================
def mutate_one(s: torch.Tensor, lo: int, bound: int, m: int, dev: str, g: torch.Generator,
               not_prob: float = 0.0):
    """Rewire an ABSOLUTE COUNT m of ONE layer's endpoints to fresh valid sources; return a patch
    entry or None. Absolute (not a fraction): a fraction of a 16K-wide layer is hundreds of
    simultaneous rewires that randomize the child -- the ES literature's 1/L-scale small steps
    (start m=1) are what make a mutant a useful neighbour of its parent.

    not_prob: with this probability a rewired endpoint is set to the SIBLING wire's current source
    instead of a random one. Since NAND(s,s)=NOT s, that collapses the gate to a NOT -- otherwise
    unreachable, because two random wires coincide with probability ~1/pool. Lets the search learn
    NOT (and inverters generally) in a single mutation."""
    m = min(m, s.shape[1])
    if m <= 0:
        return None
    idx = torch.randint(0, s.shape[1], (m,), device=dev, generator=g)               # m gate slots
    endpoint = (torch.rand(m, device=dev, generator=g) < 0.5).long()                # wire 0 or 1
    old = s[endpoint, idx].clone()
    new = torch.randint(lo, bound, (m,), device=dev, generator=g, dtype=GENE_DTYPE)
    if not_prob > 0.0:                                                               # -> NOT gate
        sibling = s[1 - endpoint, idx]                                              # other wire's src
        new = torch.where(torch.rand(m, device=dev, generator=g) < not_prob, sibling, new)
    s[endpoint, idx] = new
    return (s, endpoint, idx, old, new)


def apply_mutation(srcs, offs, m: int, dev: str, g: torch.Generator, not_prob: float = 0.0) -> list:
    """Mutate EVERY layer of `srcs` in place by m endpoints each; return a patch [(...)]."""
    patch = [mutate_one(s, lo_of(offs, l), offs[l], m, dev, g, not_prob) for l, s in enumerate(srcs)]
    return [e for e in patch if e is not None]


def restore(patch, field: int) -> None:
    """Write back a patch column: field=3 restores the OLD wiring (undo), field=4 the NEW (redo)."""
    for entry in patch:
        s, endpoint, idx = entry[0], entry[1], entry[2]
        s[endpoint, idx] = entry[field]


def best_of_k(current, Xb, yb, B, offs, k, m, dev, g, not_prob=0.0) -> tuple[float, int]:
    """One (1+lambda) step, FULL forward: score the incumbent, then k-1 mutants (each mutates every
    layer by m endpoints) branched from the SAME incumbent (undo between them), commit the best by
    smooth FITNESS (mean vote margin). Ties -> incumbent. Full forward per candidate."""
    a0 = forward_acts(current, Xb, offs)
    target = _target_packed(yb, a0.shape[1], offs[-1] - offs[-2]) if _FITNESS == "bits" else None
    best_fit, win_acc = score_cand(a0, offs, yb, B, target)
    winner = None
    for _ in range(k - 1):
        patch = apply_mutation(current, offs, m, dev, g, not_prob)
        fit, acc = score_cand(forward_acts(current, Xb, offs), offs, yb, B, target)
        restore(patch, 3)                                          # undo -> back to incumbent
        if fit > best_fit:                                        # strictly better fitness wins
            best_fit, win_acc, winner = fit, acc, patch
    if winner is not None:
        restore(winner, 4)                                        # commit the winning mutant
    return win_acc, int(winner is not None)


def best_of_k_delta(current, Xb, yb, B, offs, k, m, layers, dev, g, stats=None,
                    low_mem=False, not_prob=0.0) -> tuple[float, int]:
    """One (1+lambda) step, DELTA forward -- the >=2x path. Compute the incumbent's activations
    ONCE; each mutant rewires m endpoints of a SINGLE layer l (drawn from `layers`, biased deep so
    the tail is short) and recomputes only layers l..end IN PLACE over the cached buffer, then
    restores the tail for the next mutant. Selection is by smooth FITNESS (mean vote margin).

    low_mem: instead of cloning the incumbent tail to restore it (a second full-size activation
    buffer -- 8GB at 1e9 gates), UNDO the genome and RECOMPUTE the tail to restore acts. Trades one
    extra tail-forward per mutant for ~half the activation memory -- the difference between fitting a
    24GB card or not at billion-gate scale.

    If `stats` is given (arrays n/ns/sd length L), record per mutated-layer: attempts, #improving,
    and summed fitness delta vs the INCUMBENT -- the per-region mutation-effect diagnostic."""
    acts = forward_acts(current, Xb, offs)                          # incumbent, once
    target = _target_packed(yb, acts.shape[1], offs[-1] - offs[-2]) if _FITNESS == "bits" else None
    inc_fit, inc_acc = score_cand(acts, offs, yb, B, target)        # fixed reference for this gen
    best_fit, win_acc = inc_fit, inc_acc
    winner = None
    for l in layers.tolist():                                       # k-1 pre-drawn mutation layers
        entry = mutate_one(current[l], lo_of(offs, l), offs[l], m, dev, g, not_prob)
        if entry is None:                                          # empty draw: nothing changed
            continue
        backup = None if low_mem else acts[offs[l]:].clone()       # save tail (unless low-mem)
        forward_acts(current, Xb, offs, lstart=l, acts=acts)       # recompute only the tail
        fit, acc = score_cand(acts, offs, yb, B, target)
        current[l][entry[1], entry[2]] = entry[3]                  # undo genome (back to incumbent)
        if low_mem:
            forward_acts(current, Xb, offs, lstart=l, acts=acts)   # restore acts by recompute
        else:
            acts[offs[l]:] = backup                                # restore incumbent tail from clone
        if stats is not None:
            stats["n"][l] += 1
            stats["sd"][l] += fit - inc_fit
            stats["ns"][l] += int(fit > inc_fit)
        if fit > best_fit:
            best_fit, win_acc, winner = fit, acc, entry
    if winner is not None:
        winner[0][winner[1], winner[2]] = winner[4]               # commit the winning mutant
    return win_acc, int(winner is not None)


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
        e = mutate_one(srcs[l], 0, offs[l], 5, dev, g)             # force a real mutation
        full = forward_acts(srcs, Xp, offs)                       # ground truth
        delta = forward_acts(srcs, Xp, offs, lstart=l, acts=base.clone())
        assert torch.equal(full, delta), f"delta forward disagrees at split layer {l}!"
        srcs[l][e[1], e[2]] = e[3]                                # undo
    print(f"selftest OK on {dev}: packed == naive AND delta == full at every split layer "
          f"(B={B}, W={W}, D={D}, R={R}); votes max |diff| = {int((v_ref - v_pack).abs().max())}")


# ==========================================================================================
# Benchmark + PROFILE: time gens/sec, peak memory, and a per-stage breakdown (where the time
# goes: forward gather+NAND vs readout unpack vs mutation) on random data -- no CIFAR needed.
# ==========================================================================================
def _sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def bench(args, dev: str) -> None:
    g = torch.Generator(device=dev).manual_seed(0)
    n_in = 3 * args.num_bits * 32 * 32
    widths, offs = build_dims(n_in, args.width, args.depth, args.out_width)
    bw = max(1, args.batch // WORD)
    B = bw * WORD
    Xp = torch.randint(-(2**63), 2**63 - 1, (bw, n_in), dtype=torch.int64, device=dev)
    y = torch.randint(0, CLS, (B,), device=dev)
    cur = init_genome(widths, offs, dev, g)
    L = len(widths)
    lweights = (torch.arange(L, device=dev, dtype=torch.float32) + 1) ** args.delta_bias
    gates = args.width * args.depth + args.out_width
    genome_gb = sum(2 * w * cur[0].element_size() for w in widths) / 1e9
    print(f"bench W={args.width} D={args.depth} R={args.out_width} gates={gates:,} k={args.k} "
          f"batch={B} device={dev} | genome ~{genome_gb:.1f}GB acts ~{bw*offs[-1]*8/1e9:.1f}GB "
          f"| low_mem={args.low_mem}", flush=True)

    def full_gen():
        best_of_k(cur, Xp, y, B, offs, args.k, args.mut_count, dev, g)

    def delta_gen():
        layers = torch.multinomial(lweights, args.k - 1, replacement=True, generator=g)
        best_of_k_delta(cur, Xp, y, B, offs, args.k, args.mut_count, layers, dev, g,
                        low_mem=args.low_mem)

    def run(fn, NG):
        for _ in range(3):                                         # warmup (CUDA autotune/alloc)
            fn()
        _sync(dev)
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(NG):
            fn()
        _sync(dev)
        dt = 1000 * (time.time() - t0) / NG                        # ms/gen
        peak = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0
        return dt, peak

    NG = args.bench_gens
    d_delta, p_delta = run(delta_gen, NG)
    print(f"DELTA forward: {d_delta:7.2f} ms/gen ({1000/d_delta:5.1f} gen/s) bias={args.delta_bias}"
          + (f" | peak {p_delta:.2f}GB" if dev == "cuda" else ""), flush=True)
    if not args.low_mem:                       # full path holds k activation buffers -- skip at scale
        d_full, p_full = run(full_gen, NG)
        print(f"FULL  forward: {d_full:7.2f} ms/gen ({1000/d_full:5.1f} gen/s)"
              + (f" | peak {p_full:.2f}GB" if dev == "cuda" else ""), flush=True)
        print(f"==> DELTA speedup {d_full/d_delta:.2f}x", flush=True)

    if args.torch_profile and dev == "cuda":                       # top CUDA ops by self time
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                delta_gen()
            _sync(dev)
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=12), flush=True)


# ==========================================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/cifar-10-batches-py"))
    p.add_argument("--download", action="store_true")
    p.add_argument("--selftest", action="store_true", help="verify packed==naive and exit")
    p.add_argument("--bench", action="store_true", help="time gens/sec on random data and exit")
    p.add_argument("--bench-gens", type=int, default=200)
    p.add_argument("--torch-profile", action="store_true", help="also dump top CUDA ops (bench)")
    p.add_argument("--num-bits", type=int, default=4, help="thermometer bits per channel")
    p.add_argument("--width", type=int, default=65536, help="gates per hidden layer (W); 64K default")
    p.add_argument("--depth", type=int, default=16, help="number of hidden NAND layers (D)")
    p.add_argument("--out-width", type=int, default=2560,
                   help="gates in the thin output/readout layer (R, multiple of 10)")
    p.add_argument("--gens", type=int, default=200000, help="genetic generations (upper cap)")
    p.add_argument("--max-minutes", type=float, default=0.0,
                   help="stop after this many minutes of search (0 = no wall-clock limit)")
    p.add_argument("--k", type=int, default=16, help="candidates per generation (incl. incumbent)")
    p.add_argument("--mut-count", type=int, default=1,
                   help="ABSOLUTE number of gate endpoints a mutant rewires (per mutated layer); "
                        "small = 1/L-scale steps (ES literature), NOT a fraction of the width. "
                        "Swept best at 1 (33.6%% vs 29%% at 64) with margin fitness at batch 8192")
    p.add_argument("--not-prob", type=float, default=0.5,
                   help="prob. a rewired endpoint copies its SIBLING wire's source -> NAND(s,s)=NOT s"
                        "; makes inverters reachable in one mutation (else ~1/pool, never)")
    p.add_argument("--fitness", choices=["margin", "ce", "bits"], default="margin",
                   help="selection fitness: 'margin' = mean(correct-best_wrong votes); 'ce' = "
                        "-cross_entropy(softmax(votes/sqrt(group_size)), y); 'bits' = integer packed "
                        "point count vs the natural class code (w1 per correct-1 gate, 1 per "
                        "correct-0), no unpack/argmax -- fast + dense per-gate signal")
    p.add_argument("--bits-w1", type=float, default=float(CLS),
                   help="'bits' fitness: points for a correctly-1 gate (correct-0 = 1 point). "
                        "Default CLS balances the ~CLS:1 zero:one target ratio")
    p.add_argument("--no-delta", action="store_true",
                   help="use the full-forward GA (mutate every layer) instead of the delta forward")
    p.add_argument("--low-mem", action="store_true",
                   help="delta: restore acts by recompute instead of a tail clone (~half the "
                        "activation memory; needed for billion-gate nets on a 24GB card)")
    p.add_argument("--locality", type=int, default=0,
                   help="depth-locality: a gate may only wire to the previous K layers (0 = any "
                        "earlier signal). 2 = previous/pre-previous only -> forces signals through "
                        "the depth instead of input->output skips that make a deep net behave shallow")
    p.add_argument("--delta-bias", type=float, default=3.0,
                   help="delta forward: layer-choice weight (l+1)^bias; higher = mutate deeper "
                        "(shorter tail = faster) layers more; 0 = uniform")
    p.add_argument("--diag", action="store_true",
                   help="per-layer mutation-effect diagnostic: force UNIFORM layer sampling and log "
                        "each layer's mutation success-rate + mean fitness delta per eval window")
    p.add_argument("--batch", type=int, default=8192, help="minibatch size for fitness scoring")
    p.add_argument("--eval-every", type=int, default=200, help="gens between full val/test eval")
    p.add_argument("--out", type=Path, help="prefix for .jsonl and .pkl")
    p.add_argument("--fresh", action="store_true", help="ignore any saved .pkl and start over")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    dev = args.device

    if args.selftest:
        selftest(dev)
        return
    if args.bench:
        bench(args, dev)
        return
    if args.out is None:
        p.error("--out is required unless --selftest/--bench")
    assert args.out_width % CLS == 0, "--out-width must be a multiple of 10"
    W, D, R = args.width, args.depth, args.out_width

    # ---- DATA-PARALLEL init (torchrun sets RANK/WORLD_SIZE/LOCAL_RANK). Each rank runs the SAME GA
    #      on a different data shard; only fitness scalars are all-reduced (see score_of). ----------
    global _WORLD, _FITNESS, _BITS_W1, _LOCALITY
    _FITNESS, _BITS_W1, _LOCALITY = args.fitness, args.bits_w1, args.locality
    rank, local_rank = 0, 0
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group("nccl")
        rank, _WORLD, local_rank = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dev = f"cuda:{local_rank}"
    is_main = rank == 0

    # ---- data + thermometer bits, packed once (batch axis -> int64 words) -----------------
    tx, ty, ex, ey = load_cifar10(args.data_dir, args.download)
    nv = 5000
    enc = Thermometer(num_bits=args.num_bits).fit(tx[:2000]).to(dev)

    @torch.no_grad()
    def encode(images: torch.Tensor) -> torch.Tensor:        # (N, n_in) uint8, chunked for memory
        outs = [enc(images[i:i + 4096].to(dev)).flatten(1).to(torch.uint8)
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
    ntr_ev = min(Nw, 160)                                    # train-error on a subset (fast eval)
    Xtr_ev, Btr_ev = Xtr_p[:ntr_ev], ntr_ev * WORD
    ytr_ev = ytr[:ntr_ev * WORD]

    widths, offs = build_dims(n_in, W, D, R)
    assert offs[-1] < 2**31, f"n_signals {offs[-1]} exceeds int32 source-id range (2^31)"
    L = len(widths)
    # TWO generators: mgen (mutations+layer choice) is SEEDED IDENTICALLY on every rank so all ranks
    # propose the same mutants and stay in lockstep; dgen (batch-shard selection) is per-rank so each
    # rank scores on different data -> data parallelism. On 1 GPU they're just two independent RNGs.
    mgen = torch.Generator(device=dev).manual_seed(args.seed)
    dgen = torch.Generator(device=dev).manual_seed(args.seed + 1 + rank)
    bw = max(1, args.batch // WORD)                          # words per fitness minibatch (PER RANK)
    # delta forward: draw each mutant's single mutated layer, biased toward DEEP layers (shorter
    # tail to recompute). weight(l) = (l+1)^bias; multinomial with replacement, k-1 per generation.
    # --diag forces uniform weights so every layer gets equal attempts (fair per-region measurement).
    lweights = torch.ones(L, device=dev) if args.diag else \
        (torch.arange(L, device=dev, dtype=torch.float32) + 1) ** args.delta_bias
    diag = {"n": [0] * L, "ns": [0] * L, "sd": [0.0] * L} if args.diag else None

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    # append suffixes as STRINGS, not Path.with_suffix -- an --out like "exp_rate0.002" has a ".002"
    # that with_suffix would strip, silently collapsing distinct runs onto one file.
    ckpt = Path(str(out) + ".pkl")
    jsonl = Path(str(out) + ".jsonl")

    def save(current, gen, best_test):                       # cheap genome checkpoint (rank 0 only)
        if not is_main:
            return
        with open(ckpt, "wb") as f:
            pickle.dump({"args": vars(args) | {"out": str(out)}, "gen": gen,
                         "best_test": best_test, "srcs": [s.cpu() for s in current],
                         "thr": enc.thresholds.cpu(), "num_bits": args.num_bits,
                         "n_in": n_in, "offs": offs, "W": W, "D": D, "R": R}, f)

    # ---- resume-by-default: pick up the saved genome + jsonl if present -------------------
    if not args.fresh and ckpt.exists():
        c = pickle.load(open(ckpt, "rb"))
        current = [s.to(dev, GENE_DTYPE) for s in c["srcs"]]
        start_gen, best_test = c["gen"] + 1, c["best_test"]
        if is_main:
            print(f"resume from {ckpt} at gen {start_gen} (best_test {100*best_test:.2f})", flush=True)
    else:
        current = init_genome(widths, offs, dev, mgen)       # mgen => identical genome on every rank
        start_gen, best_test = 0, 0.0
        if is_main:
            jsonl.write_text("")

    mode = "full-forward" if args.no_delta else f"delta bias {args.delta_bias}"
    eff = bw * WORD * _WORLD
    if is_main:
        print(f"nand_ga n_in={n_in} W={W} D={D} R={R} gates={W * D + R:,} k={args.k} "
              f"mut_count={args.mut_count} not={args.not_prob} batch/rank={bw * WORD} world={_WORLD} "
              f"eff_batch={eff} device={dev} | fitness={_FITNESS} locality={_LOCALITY} | {mode} "
              f"| low_mem={args.low_mem} "
              f"| acts ~{bw * offs[-1] * 8 / 1e9:.1f}GB genome ~{sum(2*w for w in widths)*4/1e9:.1f}GB",
              flush=True)

    # ---- (1+lambda) genetic search over wirings ------------------------------------------
    t0 = time.time()
    for gen in range(start_gen, args.gens):
        wsel = torch.randint(0, Nw, (bw,), device=dev, generator=dgen)  # per-rank word-blocks (shard)
        Xb = Xtr_p[wsel]
        yb = ytr[(wsel[:, None] * WORD + torch.arange(WORD, device=dev)).reshape(-1)]

        if args.no_delta:
            batch_acc, _ = best_of_k(current, Xb, yb, bw * WORD, offs, args.k, args.mut_count, dev,
                                     mgen, not_prob=args.not_prob)
        else:
            layers = torch.multinomial(lweights, args.k - 1, replacement=True, generator=mgen)
            batch_acc, _ = best_of_k_delta(current, Xb, yb, bw * WORD, offs, args.k, args.mut_count,
                                           layers, dev, mgen, stats=diag, low_mem=args.low_mem,
                                           not_prob=args.not_prob)

        over_time = args.max_minutes > 0 and (time.time() - t0) / 60 >= args.max_minutes
        if _WORLD > 1:                                        # all ranks must stop together (else the
            flag = torch.tensor(1.0 if over_time else 0.0, device=dev)  # next all-reduce would hang)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            over_time = bool(flag.item())
        if gen % args.eval_every == 0 or gen == args.gens - 1 or over_time:
            tr = accuracy(current, Xtr_ev, ytr_ev, Btr_ev, offs)     # train-error subset
            va = accuracy(current, Xva_p, yva, Bva, offs)
            te = accuracy(current, Xte_p, yte, Bte, offs)
            best_test = max(best_test, te)
            if is_main:                                      # only rank 0 logs/checkpoints
                # accuracy AND loss (= 1 - accuracy, as %) for train/val/test -- the DD y-axis
                rec = {"gen": gen, "gates": W * D + R, "batch": round(100 * batch_acc, 2),
                       "train": round(100 * tr, 2), "val": round(100 * va, 2), "test": round(100 * te, 2),
                       "train_loss": round(100 * (1 - tr), 2), "val_loss": round(100 * (1 - va), 2),
                       "test_loss": round(100 * (1 - te), 2), "min": round((time.time() - t0) / 60, 2),
                       "gps": round((gen - start_gen + 1) / max(time.time() - t0, 1e-6), 1)}
                with open(jsonl, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                save(current, gen, best_test)                # checkpoint each eval (resume-safe)
                print(f"gen {gen:6d} | batch {rec['batch']:5.2f} | train {rec['train']:5.2f} "
                      f"val {rec['val']:5.2f} test {rec['test']:5.2f} (best {100 * best_test:5.2f}) | "
                      f"{rec['gps']:6.1f} gen/s | {rec['min']:6.2f}m", flush=True)
                if diag is not None:                         # per-layer mutation effect this window
                    parts = []
                    for l in range(L):
                        n = diag["n"][l]
                        sr = 100 * diag["ns"][l] / n if n else 0.0    # % of mutations that improve
                        md = diag["sd"][l] / n if n else 0.0          # mean fitness (margin) delta
                        parts.append(f"L{l}:{sr:4.1f}%/{md:+.3f}")
                    print("   per-layer succ%/meanΔmargin: " + "  ".join(parts), flush=True)
                    diag = {"n": [0] * L, "ns": [0] * L, "sd": [0.0] * L}  # reset window
        if over_time:
            if is_main:
                print(f"stop: hit --max-minutes {args.max_minutes}", flush=True)
            break

    save(current, gen, best_test)
    if is_main:
        print(f"saved genome ({W * D + R} gates) -> {ckpt}  best_test={100 * best_test:.2f}", flush=True)
    if _WORLD > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
