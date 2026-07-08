"""Compare forward-pass variants for the NAND net: baseline torch vs fused Triton vs torch.compile.
Verifies each is bit-identical to the baseline, then times a full forward (in place). Run on a GPU.

    python scratch_genetic/speed_probe.py --width 65536 --depth 16 --batch 8192
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nand_ga as N  # noqa: E402
from nand_kernels import forward_acts_sig_triton, forward_acts_triton  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=65536)
    p.add_argument("--depth", type=int, default=16)
    p.add_argument("--out-width", type=int, default=100)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    dev = args.device
    n_in = 3 * 4 * 32 * 32
    widths, offs = N.build_dims(n_in, args.width, args.depth, args.out_width)
    bw = max(1, args.batch // N.WORD)
    g = torch.Generator(device=dev).manual_seed(0)
    Xp = torch.randint(-(2**63), 2**63 - 1, (bw, n_in), dtype=torch.int64, device=dev)
    srcs = N.init_genome(widths, offs, dev, g)
    print(f"probe W={args.width} D={args.depth} R={args.out_width} gates={args.width*args.depth:,} "
          f"batch={bw*N.WORD} (Bw={bw}) device={dev}", flush=True)

    def sync():
        torch.cuda.synchronize()

    # ---- signal-major forward: acts is (n_sig, Bw) so each gathered signal is a CONTIGUOUS row
    #      (coalesced reads) instead of a strided column. index_select on dim 0. -----------------
    def forward_sigmajor(srcs, Xp, offs, acts=None):
        Bw, n_in = Xp.shape
        if acts is None:
            acts = torch.empty(offs[-1], Bw, dtype=torch.int64, device=Xp.device)
            acts[:n_in] = Xp.t()
        for l, s in enumerate(srcs):
            w = s.shape[1]
            a = acts.index_select(0, s[0].long())
            b = acts.index_select(0, s[1].long())
            acts[offs[l]:offs[l + 1]] = ~(a & b)
        return acts

    # ---- correctness: triton + signal-major must match baseline bit-for-bit -----------------
    ref = N.forward_acts(srcs, Xp, offs)
    tri = forward_acts_triton(srcs, Xp, offs)
    sig = forward_sigmajor(srcs, Xp, offs)
    sigt = forward_acts_sig_triton(srcs, Xp, offs)
    print(f"triton=={torch.equal(ref, tri)} sigmajor=={torch.equal(ref, sig.t().contiguous())} "
          f"sig_triton=={torch.equal(ref, sigt.t().contiguous())}", flush=True)

    # ---- torch.compile variants ------------------------------------------------------------
    def full_fwd(srcs, Xp):
        return N.forward_acts(srcs, Xp, offs)

    results = {}

    def timeit(name, fn, acts_buf):
        for _ in range(5):                                   # warmup (compile / autotune)
            fn(acts_buf)
        sync()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(args.iters):
            fn(acts_buf)
        sync()
        ms = 1000 * (time.time() - t0) / args.iters
        peak = torch.cuda.max_memory_allocated() / 1e9
        results[name] = ms
        print(f"{name:22s} {ms:8.3f} ms/fwd   peak {peak:5.2f}GB", flush=True)

    acts = torch.empty(bw, offs[-1], dtype=torch.int64, device=dev)
    acts[:, :n_in] = Xp
    actsT = torch.empty(offs[-1], bw, dtype=torch.int64, device=dev)
    actsT[:n_in] = Xp.t()
    # in-place recompute (lstart=0) so we time compute, not allocation
    timeit("baseline (torch)", lambda a: N.forward_acts(srcs, Xp, offs, lstart=0, acts=a), acts)
    timeit("triton (fused)", lambda a: forward_acts_triton(srcs, Xp, offs, lstart=0, acts=a), acts)
    timeit("sigmajor (torch)", lambda a: forward_sigmajor(srcs, Xp, offs, acts=a), actsT)
    timeit("sig_triton (fused)", lambda a: forward_acts_sig_triton(srcs, Xp, offs, acts=a), actsT)

    base = results.get("baseline (torch)", float("nan"))
    print("\n=== speedups vs baseline ===", flush=True)
    for k, v in results.items():
        print(f"  {k:22s} {base/v:5.2f}x", flush=True)


if __name__ == "__main__":
    main()
