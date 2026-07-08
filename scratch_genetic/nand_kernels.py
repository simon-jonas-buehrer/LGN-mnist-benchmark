"""Fused Triton kernel for the packed NAND forward -- the big-restructure speed win.

The pure-torch layer does FOUR memory-heavy ops: index_select gathers both wires into a (Bw, 2W)
int64 intermediate, then bitwise_and, bitwise_not, and a strided slice-store. That materialises 2W
extra columns and launches ~4 kernels per layer. This kernel fuses all of it: each program loads
the two source words for its gates directly, computes ~(a & b), and stores -- 2 reads + 1 write per
gate, no intermediate, ONE launch per layer. Also removes the 2W int64 scratch (1GB at 1e9 gates).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _nand_layer_kernel(acts_ptr, s0_ptr, s1_ptr, n_sig, W, off, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)                      # batch word (row of acts)
    g = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)       # gate ids in this block
    mask = g < W
    base = row * n_sig                                       # start of this row (int64 addressing)
    s0 = tl.load(s0_ptr + g, mask=mask, other=0).to(tl.int64)
    s1 = tl.load(s1_ptr + g, mask=mask, other=0).to(tl.int64)
    a = tl.load(acts_ptr + base + s0, mask=mask, other=0)    # gather wire 0 (int64 word = 64 samples)
    b = tl.load(acts_ptr + base + s1, mask=mask, other=0)    # gather wire 1
    tl.store(acts_ptr + base + off + g, ~(a & b), mask=mask)  # 64-way NAND, in place


def nand_layer_triton(acts: torch.Tensor, s: torch.Tensor, offs, l: int) -> None:
    """Fused in-place NAND for layer l. `acts` (Bw, n_sig) int64 contiguous; `s` (2, W) int32."""
    Bw, n_sig = acts.shape
    W = s.shape[1]
    BLOCK = 1024
    grid = (Bw, triton.cdiv(W, BLOCK))
    _nand_layer_kernel[grid](acts, s[0], s[1], n_sig, W, offs[l], BLOCK=BLOCK)


@torch.no_grad()
def forward_acts_triton(srcs, Xp: torch.Tensor, offs, lstart: int = 0,
                        acts: torch.Tensor | None = None) -> torch.Tensor:
    """Same contract as nand_ga.forward_acts but every layer runs the fused Triton kernel."""
    if acts is None:
        Bw, n_in = Xp.shape
        acts = torch.empty(Bw, offs[-1], dtype=torch.int64, device=Xp.device)
        acts[:, :n_in] = Xp
    for l in range(lstart, len(srcs)):
        nand_layer_triton(acts, srcs[l].contiguous(), offs, l)
    return acts


# ==========================================================================================
# SIGNAL-MAJOR fused kernel: acts is (n_sig, Bw). A gate reads two CONTIGUOUS Bw-wide rows
# (coalesced) and writes one -- combines the coalescing win (row gather) with fusion (no a,b
# intermediate). Each program handles a GB x BB tile of (gates, batch-words).
# ==========================================================================================
@triton.jit
def _nand_sig_kernel(acts_ptr, s0_ptr, s1_ptr, Bw, W, off, GB: tl.constexpr, BB: tl.constexpr):
    g = tl.program_id(0) * GB + tl.arange(0, GB)            # gate ids
    bc = tl.program_id(1) * BB + tl.arange(0, BB)           # batch-word ids
    gm, bm = g < W, bc < Bw
    s0 = tl.load(s0_ptr + g, mask=gm, other=0).to(tl.int64)[:, None]
    s1 = tl.load(s1_ptr + g, mask=gm, other=0).to(tl.int64)[:, None]
    col = bc[None, :]
    m = gm[:, None] & bm[None, :]
    a = tl.load(acts_ptr + s0 * Bw + col, mask=m, other=0)  # (GB, BB), coalesced along batch
    b = tl.load(acts_ptr + s1 * Bw + col, mask=m, other=0)
    tl.store(acts_ptr + (off + g[:, None]) * Bw + col, ~(a & b), mask=m)


def nand_layer_sig_triton(acts: torch.Tensor, s: torch.Tensor, offs, l: int) -> None:
    n_sig, Bw = acts.shape
    W = s.shape[1]
    GB, BB = 32, 32
    grid = (triton.cdiv(W, GB), triton.cdiv(Bw, BB))
    _nand_sig_kernel[grid](acts, s[0], s[1], Bw, W, offs[l], GB=GB, BB=BB)


@torch.no_grad()
def forward_acts_sig_triton(srcs, Xp: torch.Tensor, offs, lstart: int = 0,
                            acts: torch.Tensor | None = None) -> torch.Tensor:
    """Signal-major (n_sig, Bw) fused forward. Xp is (Bw, n_in) row-major -> transpose the input."""
    if acts is None:
        Bw, n_in = Xp.shape
        acts = torch.empty(offs[-1], Bw, dtype=torch.int64, device=Xp.device)
        acts[:n_in] = Xp.t()
    for l in range(lstart, len(srcs)):
        nand_layer_sig_triton(acts, srcs[l].contiguous(), offs, l)
    return acts
