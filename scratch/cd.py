"""Pure coordinate descent on a huge, fixed-size, PURELY RANDOM stack of HASH-GATE layers
-- learnable levers per gate: CONNECTIONS, TABLE, HASH WEIGHTS, and per-dim WEIGHT SHARING.

No building, no growing, no backprop, no op history. The image stays a 3D grid
(Ci=3*num_bits thermometer channels, 32, 32); the model is a stack of L windows
(C_l, H_l, H_l), --channels "c0,c1,..." x --spatial "h0,h1,..." (one channel number = the
original flat depth-0 model). Coarser upper layers are CNN-style pooling pyramids: fewer
slots, bigger effective receptive fields, and far less stored-output memory (slots x D
bits), which is what buys scale. A gate in layer l taps ANY lower source: the input grid
or the outputs of layers < l (skip connections included) -- rank is explicit, so shifted
reads can never form a cycle. Head: every slot of every layer votes, slot class = global
channel % 10, score_c = popcount / sqrt(S/10). Loss: Crammer-Singer hinge on the FULL
train set -- every accept is exact on all of train.

A gate is a HASH GATE -- one gate type subsuming all function families: K connections
(absolute source coords) + K learned integer hash weights c_k + an M-bit table T; output
= T[(sum_k c_k * x_k) mod M] (--tsize sets M; default = the classic 2**K coupling; per-gate
M growth is stage 2b). K and M are independent budgets -- taps vs table bits -- so K > 8
works with small tables (--fan-in 16 --tsize 64 --gate hash). Corners of
the (c, T) space: c_k = 2**k is the classic full LUT (--gate lut, bit-exact with the old
executor); c in {-1,0,1} with a threshold-step T is a BitNet-style ternary threshold gate
(--gate ternary, the stable no-wrap corner). CD moves the weights themselves (cd-cf), so
fan-in and function family are earned, not chosen: c_k=0 makes a tap inert (learned
fan-in), and nothing but measured hinge decides where between threshold, symmetric, LUT
and hashed a gate lives. Each gate also has three sharing degrees (num_copies per window
dimension, powers of 2, default 1 = unshared) + three
learned input STRIDES (step per dim). A gate with copies (nc,nh,nw) occupies the slots of
ITS layer strided dim/n from its base (mod dim) -- the output tiling is fixed -- but copy
(i,j,k) reads its connections shifted by (i*step_c, j*step_h, k*step_w) within each tap's
own source grid (spatial shifts scale with the grid-size ratio, mod that grid's dims).
step = dim/n is plain conv striding; smaller overlaps receptive fields, larger dilates,
0 ties all copies to identical inputs. Sharing IS convolution -- kernel shape (the taps),
stride, dilation and tying all per-gate and learnable.

Depth costs exactly one thing: an accept in layer l changes stored output bits that feed
higher layers, so its EXACT hinge effect includes a CASCADE -- every reader of a dirty
source is recomputed on all samples, its vote delta added, its own changed bits propagated
further up, and everything XOR-reverted if the package does not improve. A top-layer accept
cascades nowhere and costs the same as depth 0; a bottom-layer accept can touch everything
above (--casc-cap bounds it). The bandit in the main loop measures hinge-per-second per
(operator, layer), so cheap top-layer moves and expensive-but-fertile bottom-layer moves
are traded off automatically.

Why it is fast:
  * per-slot output bits are stored ONCE, bit-packed (slots x D/8 bytes), so evaluating a
    gate is a gather, never a recursive recompute;
  * each (copy, sample) lands on exactly ONE of the M table cells (any hash weights), so
    cells partition the (copies x samples) rows and one visit sets ALL M bits at once --
    block-CD over the whole table for the price of one evaluation (the cell benefit is the
    exact direct-vote delta; the cascade is verified at accept time);
  * a connection move scores n random replacement sources exactly (same delta tables) over
    all copies and takes the best; removing tap k from the hash is one subtract
    (bas = h - c_k*x_k mod M), so candidate sources AND candidate weights are each one
    add away -- the same shifted-gather trick prices both levers.

    .venv/bin/python scratch/cd.py --device cuda --channels 640,320,160,80,40,40,20,20
    bash scratch/cd.sh                                              # inside an srun GPU shell
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Thermometer  # noqa: E402
from train import load_cifar10  # noqa: E402

CLS = 10


# ---- bitpacking: a signal is one bit per image, stored as ceil(D/64) int64 words ----------
def _shifts(device) -> torch.Tensor:
    return torch.arange(64, dtype=torch.int64, device=device)


def pack_bits(bits: torch.Tensor, row_chunk: int = 1024) -> torch.Tensor:
    """(n, D) of {0,1} -> (n, W) int64, in row chunks so the int64 temporary stays small."""
    n, d = bits.shape
    w = (d + 63) // 64
    if w * 64 - d:
        bits = torch.cat([bits, bits.new_zeros(n, w * 64 - d)], dim=1)
    out = torch.empty((n, w), dtype=torch.int64, device=bits.device)
    for r0 in range(0, n, row_chunk):
        b = bits[r0:r0 + row_chunk]
        out[r0:r0 + b.shape[0]] = (b.to(torch.int64).view(-1, w, 64) << _shifts(bits.device)).sum(-1)
    return out


def unpack_bits(words: torch.Tensor, d: int, word_chunk: int = 32) -> torch.Tensor:
    """(n, W) int64 -> (n, D) uint8, in word chunks so the int64 temporary stays small."""
    n, w = words.shape
    sh = _shifts(words.device)
    out = torch.empty((n, w * 64), dtype=torch.uint8, device=words.device)
    for c0 in range(0, w, word_chunk):
        wc = words[:, c0:c0 + word_chunk]
        out[:, c0 * 64:(c0 + wc.shape[1]) * 64] = (((wc.unsqueeze(-1) >> sh) & 1)
                                                   .to(torch.uint8).reshape(n, -1))
    return out[:, :d]


# ==========================================================================================
class Win:
    """A stack of (C_l, H_l, H_l) windows of LUT-gate copies over a (Ci, HI, HI) binary
    input grid.

    Source coordinate space (what `conn` indexes): rows [0, N) are the input bits; N + s is
    the stored output of slot s. Layers own contiguous slot ranges, in order, so "layer l
    may read sources < N + cum_slots[l]" is a single bound. All grids are square; spatial
    shifts translate across grids by the size ratio (all sizes are powers of two)."""

    EPS = 1e-4  # accept threshold on the (tau-scaled) full-train hinge decrease

    def __init__(self, ci: int, hw: int, chs: list[int], hws: list[int], fan_in: int,
                 max_copies: int, device: str, init_deg: tuple[int, int, int] = (0, 0, 0),
                 init_loc: int = 0, init_res: float = 0.0, gate: str = "lut",
                 tsize: int = 0):
        assert len(hws) == len(chs) and all(c % CLS == 0 for c in chs)
        assert all(h & (h - 1) == 0 and 4 <= h <= hw for h in hws)   # powers of two
        self.init_loc, self.init_res, self.gate = init_loc, init_res, gate
        self.chs, self.hws, self.L = list(chs), list(hws), len(chs)
        self.Ci, self.HWI = ci, hw                        # input grid
        self.N = ci * hw * hw                             # input bits
        self.S = sum(c * h * h for c, h in zip(chs, hws)) # slots over all layers (= votes)
        # table size M decouples from fan-in K: eval is O(K) unpacks, storage O(M) bits --
        # K is a TAP BUDGET (c=0 taps are inert), M an EXPRESSIVITY budget. --tsize 0
        # keeps the classic coupling M = 2**K (required for the lut corner's c_k = 2**k).
        self.K = fan_in
        self.M = self.TT = tsize if tsize else 1 << fan_in
        assert self.M <= 16384                            # 2M-2 must fit the int16 hash acc
        if gate == "lut":
            assert fan_in <= 8 and self.M == 1 << fan_in  # place values need the full table
        if gate == "ternary":
            assert self.M >= 2 * fan_in + 2               # signed sums must stay distinct
        assert fan_in <= 8 or tsize                       # K>8 only with an explicit M
        self.tau = math.sqrt(self.S / CLS)
        self.max_copies = max_copies
        self.device = device
        self.cap = 1 << 30                                # cascade row budget (set by main)
        cum_c, cum_s = [0], [0]
        for c, h in zip(chs, hws):
            cum_c.append(cum_c[-1] + c)
            cum_s.append(cum_s[-1] + c * h * h)
        self.cum_ch = torch.tensor(cum_c, dtype=torch.long, device=device)
        self.cum_slots = torch.tensor(cum_s, dtype=torch.long, device=device)
        self.chs_t = torch.tensor(chs, dtype=torch.long, device=device)
        self.hws_t = torch.tensor(hws, dtype=torch.long, device=device)
        self.src_bound = self.N + self.cum_slots[:-1]     # tap coord -> its source grid
        self.maxdeg_c = [max(p for p in range(15) if c % (1 << p) == 0) for c in chs]
        self.maxdeg_s = [max(p for p in range(15) if h % (1 << p) == 0) for h in hws]
        # random gates tiling each layer at the initial sharing degrees (log2 copies per dim;
        # (0,0,0) = one unshared gate per slot). CD can unshare/reshare from there.
        dc, dh, dw = init_deg
        n = 1 << (dc + dh + dw)
        assert all(c % (1 << dc) == 0 for c in chs) and n <= max_copies
        assert all(h % (1 << dh) == 0 and h % (1 << dw) == 0 for h in hws)
        self.base = torch.zeros(self.S, dtype=torch.int32, device=device)
        self.conn = torch.randint(self.N, (self.S, fan_in), dtype=torch.int32, device=device)
        self.tt, self.coef = self._rand_fn(self.S)
        self.deg = torch.zeros((self.S, 3), dtype=torch.int8, device=device)
        self.step = torch.zeros((self.S, 3), dtype=torch.int16, device=device)
        self.sgn = torch.ones(self.S, dtype=torch.int8, device=device)  # vote polarity
        self.ocls = torch.zeros(self.S, dtype=torch.int8, device=device)  # learned output class
        self.alive = torch.zeros(self.S, dtype=torch.bool, device=device)
        self.owner = torch.zeros(self.S, dtype=torch.int32, device=device)
        g0 = 0
        for l, (c, hwl) in enumerate(zip(chs, hws)):
            stc, sth, stw = c >> dc, hwl >> dh, hwl >> dw # fundamental-domain dims
            ngl = (c * hwl * hwl) >> (dc + dh + dw)
            g = torch.arange(ngl, device=device)
            bc, by, bx = g // (sth * stw), (g // stw) % sth, g % stw
            self.base[g0:g0 + ngl] = (cum_s[l] + (bc * hwl + by) * hwl + bx).to(torch.int32)
            self.deg[g0:g0 + ngl] = torch.tensor(init_deg, dtype=torch.int8, device=device)
            self.step[g0:g0 + ngl] = torch.tensor([c >> dc, hwl >> dh, hwl >> dw],
                                                  dtype=torch.int16, device=device)
            self.alive[g0:g0 + ngl] = True
            self.conn[g0:g0 + ngl] = self._rand_conn(l, by, bx)
            s = torch.arange(cum_s[l + 1] - cum_s[l], device=device)
            cc, y, x = s // (hwl * hwl), (s // hwl) % hwl, s % hwl
            self.owner[cum_s[l]:cum_s[l + 1]] = (g0 + ((cc % stc) * sth + (y % sth)) * stw
                                                 + (x % stw)).to(torch.int32)
            self.ocls[cum_s[l]:cum_s[l + 1]] = ((cum_c[l] + cc) % CLS).to(torch.int8)
            g0 += ngl
        self._smask = torch.zeros(self.S, dtype=torch.bool, device=device)   # scratch
        self._sidx = torch.zeros(self.S, dtype=torch.long, device=device)    # scratch

    def set_train(self, input_bits: torch.Tensor, y: torch.Tensor, rows: int = 2048) -> None:
        self.D = input_bits.shape[1]
        self.rows = rows
        pk = pack_bits(input_bits.to(self.device))
        if self.L == 1:                                   # depth 0: no stored outputs at all
            self.src = pk
        else:                                             # reuse the buffer across re-rolls:
            shape = (self.N + self.S, pk.shape[1])        # src is half the GPU at scale, and
            if getattr(self, "src", None) is None or self.src.shape != shape:  # every slot
                self.src = torch.zeros(shape, dtype=torch.int64, device=self.device)  # row is
            self.src[:self.N] = pk                        # rewritten by the forward below
        self.y = y.to(self.device)
        self._ar = torch.arange(self.D, device=self.device)
        self.score = self.forward(self.src, self.D, rows)  # (CLS, D) exact int counts
        self.hval = self._hinge(self.score)                # cached exact hinge; every
                                                           # accept keeps it in sync
    # -- fresh-gate distributions (init and refills draw from the same mix) ---------------
    def _rand_tt(self, n: int) -> torch.Tensor:
        """Random truth tables; a fraction starts as PASS-THROUGH of tap 0 (discrete
        residual initialization, after conv-difflogic: identity gates keep signal flowing
        through depth from round 0 -- block-CD overwrites them wherever something better
        exists, so this only changes the starting basin). A/B measured: with local taps,
        beat uniform-random init 36.5 vs 31.8 val at equal budget. (PBIL-style sampling
        from surviving-gate bit marginals was A/B'd and refuted: no gain over uniform.)"""
        tt = torch.randint(0, 2, (n, self.TT), dtype=torch.int8, device=self.device).bool()
        if self.init_res > 0 and n:
            r = torch.rand(n, device=self.device) < self.init_res
            tt[r] = (torch.arange(self.TT, device=self.device) & 1) > 0
        return tt

    def _rand_fn(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Fresh gate FUNCTIONS = (table, hash weights), drawn at the requested corner of
        the hash-gate space. --gate lut: c_k = 2**k + the random/residual tables above --
        today's LUT gate, bit-exact. --gate hash: c uniform in [1, M) + random tables --
        input patterns spread over ALL M cells from round 0 (Bloom/WiSARD-style), full
        capacity at any K/M ratio; residual gates are exact tap-0 pass-throughs
        (c = (c0, 0, ..., 0), T = [j == c0]). --gate ternary: c in {-1,0,1} (stored mod
        M) and T a threshold step on the SIGNED cell value (BitNet-style ternary
        threshold, the stable no-wrap corner: |sum| <= K < M/2, so nothing aliases);
        residual = tap-0 pass-through (c = (1,0,...,0), step at 1). Measured (hg1): the
        ternary corner starts with only 2K+1 of M cells reachable -- dead capacity, badly
        behind lut. CD walks any gate anywhere in the space from any corner (cd-cf)."""
        if self.gate == "lut":
            coef = (2 ** torch.arange(self.K, device=self.device)).to(torch.int16) \
                .expand(n, self.K).contiguous()
            return self._rand_tt(n), coef
        if self.gate == "hash":
            coef = torch.randint(1, self.M, (n, self.K), device=self.device)
            tt = torch.randint(0, 2, (n, self.TT), dtype=torch.int8,
                               device=self.device).bool()
            if self.init_res > 0 and n:
                r = torch.rand(n, device=self.device) < self.init_res
                c0 = coef[:, 0]
                coef[r, 1:] = 0
                tt[r] = torch.arange(self.M, device=self.device)[None] == c0[r][:, None]
            return tt, coef.to(torch.int16)
        coef = torch.randint(-1, 2, (n, self.K), device=self.device)
        j = torch.arange(self.M, device=self.device)
        sj = torch.where(j > self.M // 2, j - self.M, j)             # signed cell value
        th = torch.randint(1 - self.K, self.K + 1, (n, 1), device=self.device)
        tt = sj[None] >= th
        if self.init_res > 0 and n:
            r = torch.rand(n, device=self.device) < self.init_res
            coef[r] = 0
            coef[r, 0] = 1
            tt[r] = sj[None] >= 1
        return tt, coef.remainder_(self.M).to(torch.int16)

    def _rand_conn(self, l: int, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Random taps for layer-l gates at spatial (y, x) (in layer-l grid units): uniform
        over all allowed sources, or -- with locality init -- within +-init_loc pixels of
        the gate's own position mapped into a random allowed source grid (the CNN locality
        prior; rewiring can undo it wherever long range pays)."""
        n = y.numel()
        if not self.init_loc:
            return torch.randint(self.N + int(self.cum_slots[l]), (n, self.K),
                                 dtype=torch.int32, device=self.device)
        offs = torch.cat([torch.zeros(1, dtype=torch.long, device=self.device),
                          self.src_bound])
        cgs = torch.cat([torch.tensor([self.Ci], device=self.device), self.chs_t])
        hgs = torch.cat([torch.tensor([self.HWI], device=self.device), self.hws_t])
        m = torch.randint(l + 1, (n, self.K), device=self.device)
        off, cg, hg = offs[m], cgs[m], hgs[m]
        c = (torch.rand(n, self.K, device=self.device) * cg).long()
        r, hwl = self.init_loc, self.hws[l]
        yy = (y[:, None] * hg // hwl
              + torch.randint(-r, r + 1, (n, self.K), device=self.device)) % hg
        xx = (x[:, None] * hg // hwl
              + torch.randint(-r, r + 1, (n, self.K), device=self.device)) % hg
        return (off + (c * hg + yy) * hg + xx).to(torch.int32)

    # -- coordinate helpers ----------------------------------------------------------------
    def _cyx(self, slot: torch.Tensor):
        """slot id -> (layer, global channel, y, x, layer grid size)."""
        lay = torch.searchsorted(self.cum_slots, slot, right=True) - 1
        hwl = self.hws_t[lay]
        loc = slot - self.cum_slots[lay]
        return lay, self.cum_ch[lay] + loc // (hwl * hwl), (loc // hwl) % hwl, loc % hwl, hwl

    def _cls(self, slot: torch.Tensor) -> torch.Tensor:
        return self.ocls[slot].long()               # learned output class (init = channel%CLS)

    def _lay(self, gid: torch.Tensor) -> torch.Tensor:
        return self._cyx(self.base[gid].long())[0]

    def _rows(self, gid: torch.Tensor):
        """Enumerate all copies of the given gates. Returns per-row: gate index into `gid`,
        window slot, input shifts (sc, sy, sx = copy index x learned step, in the gate's
        own grid units; resolved per source grid in _shift), the gate's grid size, and the
        copy index per dimension."""
        dg = self.deg[gid].long()                                    # (B, 3)
        n = 1 << dg.sum(1)
        row_gate = torch.repeat_interleave(torch.arange(gid.numel(), device=self.device), n)
        r = torch.arange(int(n.sum()), device=self.device) - (n.cumsum(0) - n)[row_gate]
        nc, nh, nw = (1 << dg[:, 0])[row_gate], (1 << dg[:, 1])[row_gate], (1 << dg[:, 2])[row_gate]
        ic, ih, iw = r // (nh * nw), (r // nw) % nh, r % nw
        lay, bc, by, bx, hwl = self._cyx(self.base[gid].long()[row_gate])
        cl = self.chs_t[lay]                                         # copies stay in-layer
        c2 = (bc - self.cum_ch[lay] + ic * (cl // nc)) % cl
        yy = (by + ih * (hwl // nh)) % hwl
        xx = (bx + iw * (hwl // nw)) % hwl
        slot = self.cum_slots[lay] + (c2 * hwl + yy) * hwl + xx
        st = self.step[gid].long()[row_gate]                         # learned INPUT strides;
        return row_gate, slot, ic * st[:, 0], (ih * st[:, 1]) % hwl, \
            (iw * st[:, 2]) % hwl, hwl, torch.stack([ic, ih, iw], 1)  # output tiling is fixed

    def _shift(self, flat: torch.Tensor, sc, sy, sx, hwl) -> torch.Tensor:
        """Shift tap coords (R, K) by per-row copy offsets, each within its OWN source grid
        (input grid or one lower layer): spatial shifts are given in the gate's grid units
        (hwl) and scale by the grid-size ratio; the channel shift wraps within the tap's
        grid -- conv semantics per source, across resolutions."""
        grid = torch.searchsorted(self.src_bound, flat, right=True)  # 0=input, m+1=layer m
        g1 = (grid - 1).clamp(min=0)
        off = torch.where(grid == 0, 0, self.src_bound[g1])
        cg = torch.where(grid == 0, self.Ci, self.chs_t[g1])
        hg = torch.where(grid == 0, self.HWI, self.hws_t[g1])
        loc = flat - off
        kc, ky, kx = loc // (hg * hg), (loc // hg) % hg, loc % hg
        kc = (kc + sc[:, None]) % cg
        ky = (ky + sy[:, None] * hg // hwl[:, None]) % hg
        kx = (kx + sx[:, None] * hg // hwl[:, None]) % hg
        return off + (kc * hg + ky) * hg + kx

    def _cells(self, flat: torch.Tensor, coef: torch.Tensor, src: torch.Tensor,
               d: int) -> torch.Tensor:
        """(R, K) source coords + (R, K) hash weights -> (R, D) table cell per (row,
        sample): cell = (sum_k c_k * x_k) mod M. The LUT corner (c_k = 2**k) reproduces
        the classic K-bit address exactly; eval is O(K) adds whatever the table size, so
        neither fan-in nor expressivity is tied to 2**K anymore. Weights are stored
        canonically in [0, M) (-1 == M-1); the running remainder keeps the accumulator
        tiny and overflow-free."""
        acc = torch.zeros((flat.shape[0], d), dtype=torch.int16, device=self.device)
        for i in range(self.K):
            acc.add_(coef[:, i:i + 1] * unpack_bits(src[flat[:, i]], d)).remainder_(self.M)
        return acc

    def _near(self, cur: torch.Tensor, radius: int, p_global: float, hi: int) -> torch.Tensor:
        """Source coords near the given ones: same source grid, same spatial neighborhood
        (+-radius in that grid's pixels, wrapped), any channel of that grid; with prob
        p_global an unrestricted jump over all sources allowed for the gate's layer (< hi)."""
        n = cur.shape[0]
        grid = torch.searchsorted(self.src_bound, cur, right=True)
        g1 = (grid - 1).clamp(min=0)
        off = torch.where(grid == 0, 0, self.src_bound[g1])
        cg = torch.where(grid == 0, self.Ci, self.chs_t[g1])
        hg = torch.where(grid == 0, self.HWI, self.hws_t[g1])
        loc = cur - off
        y, x = (loc // hg) % hg, loc % hg
        c = torch.minimum((torch.rand(n, device=self.device) * cg).long(), cg - 1)
        yy = (y + torch.randint(-radius, radius + 1, (n,), device=self.device)) % hg
        xx = (x + torch.randint(-radius, radius + 1, (n,), device=self.device)) % hg
        near = off + (c * hg + yy) * hg + xx
        glob = torch.randint(hi, (n,), device=self.device)
        return torch.where(torch.rand(n, device=self.device) < p_global, glob, near)

    def _chunks(self, gid: torch.Tensor, rows: int):
        """Split a gate list into chunks of ~`rows` total copies (ragged-safe)."""
        n = 1 << self.deg[gid].long().sum(1)
        cut = ((n.cumsum(0) - 1) // rows)
        edges = torch.cat([torch.tensor([0], device=self.device),
                           (cut[1:] != cut[:-1]).nonzero().flatten() + 1,
                           torch.tensor([gid.numel()], device=self.device)]).cpu()
        for a, b in zip(edges[:-1].tolist(), edges[1:].tolist()):
            if b > a:
                yield gid[a:b]

    # -- forward -------------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, src: torch.Tensor, d: int, rows: int = 2048) -> torch.Tensor:
        """Layer-ordered full pass: fills every slot's output row of `src` (L > 1) and
        returns the class scores."""
        score = torch.zeros((CLS, d), dtype=torch.float32, device=self.device)
        ids = self.alive.nonzero().flatten()
        lay = self._lay(ids)
        for l in range(self.L):
            for seg in self._chunks(ids[lay == l], rows):
                rg, slot, sc, sy, sx, hwl, _ = self._rows(seg)
                cells = self._cells(self._shift(self.conn[seg].long()[rg], sc, sy, sx, hwl),
                                    self.coef[seg][rg], src, d)
                out = self.tt[seg][rg].gather(1, cells.long())
                score.index_add_(0, self._cls(slot),
                                 out.float() * self.sgn[seg].float()[rg, None])
                if self.L > 1:
                    src[self.N + slot] = pack_bits(out.to(torch.uint8))
        return score

    @torch.no_grad()
    def evaluate(self, input_bits: torch.Tensor, y: torch.Tensor, rows: int = 2048) -> float:
        d = input_bits.shape[1]
        pk = pack_bits(input_bits.to(self.device))
        if self.L > 1:
            pk = torch.cat([pk, torch.zeros((self.S, pk.shape[1]), dtype=torch.int64,
                                            device=self.device)])
        s = self.forward(pk, d, rows)
        return 100.0 * (s.argmax(0).cpu() == y).float().mean().item()

    def train_acc(self) -> float:
        return 100.0 * (self.score.argmax(0) == self.y).float().mean().item()

    # -- per-sample hinge tables -----------------------------------------------------------
    def _hinge(self, score: torch.Tensor) -> float:
        L = score / self.tau
        sy = L[self.y, self._ar]
        L2 = L.clone(); L2[self.y, self._ar] = -1e9
        return float(torch.clamp(1.0 + L2.max(0).values - sy, min=0).sum())

    def _base_tables(self):
        """h0: per-sample hinge now. tup/tdn[c, s]: EXACT hinge change if class c's score on
        sample s moves by +1/-1 (one copy's output flipping)."""
        L = self.score / self.tau
        y, ar = self.y, self._ar
        sy = L[y, ar]
        oth = L.clone(); oth[y, ar] = -1e9
        m1, am1 = oth.max(0)
        oth[am1, ar] = -1e9
        m2 = oth.max(0).values
        h0 = torch.clamp(1.0 + m1 - sy, min=0)
        inv = 1.0 / self.tau
        cid = torch.arange(CLS, device=self.device)[:, None]
        isy, ism = cid == y[None], cid == am1[None]
        mo = torch.where(isy, m1[None], torch.maximum(m1[None], L + inv))
        s2 = torch.where(isy, (sy + inv)[None], sy[None].expand(CLS, -1))
        tup = torch.clamp(1.0 + mo - s2, min=0) - h0[None]
        mo = torch.where(isy, m1[None],
                         torch.where(ism, torch.maximum(m2[None], L - inv), m1[None]))
        s2 = torch.where(isy, (sy - inv)[None], sy[None].expand(CLS, -1))
        tdn = torch.clamp(1.0 + mo - s2, min=0) - h0[None]
        return h0, tup, tdn

    # -- the one exact accept test: direct votes + full cascade, apply or revert -----------
    @torch.no_grad()
    def _commit(self, slot: torch.Tensor, old: torch.Tensor, new: torch.Tensor,
                thr: float, wo: torch.Tensor | None = None,
                wn: torch.Tensor | None = None) -> bool:
        """Propose new outputs `new` for `slot` (all in ONE layer; `old` = current outputs,
        both (R, D) bool). Computes the EXACT total hinge effect: direct vote deltas plus
        the full cascade -- every reader of a dirty source (any higher layer, skip
        connections included) is recomputed on all samples, its vote delta added, its own
        changed bits propagated further. Changes are applied to src as we go so recomputes
        see the new state; if the cascade exceeds `cap` recomputed rows or the exact hinge
        does not clear `thr`, every touched row is XOR-reverted. A top-layer package
        cascades nowhere and costs the same as depth 0. wo/wn: per-row vote polarity of
        the old/new owners (None = +1); sign-only moves pass old==new and just reweight."""
        ds = torch.zeros_like(self.score)
        if wo is None:
            ds.index_add_(0, self._cls(slot), new.float() - old.float())
        else:
            ds.index_add_(0, self._cls(slot),
                          new.float() * wn[:, None] - old.float() * wo[:, None])
        applied, ok = [], True
        if self.L > 1:
            diff = pack_bits((old ^ new).to(torch.uint8))
            m = (diff != 0).any(1)
            dirty = [slot[m]]
            if int(m.sum()):
                self.src[self.N + slot[m]] ^= diff[m]
                applied.append((self.N + slot[m], diff[m]))
            l0 = int(torch.searchsorted(self.cum_slots, int(slot[0]), right=True)) - 1
            left = self.cap
            ids = self.alive.nonzero().flatten()
            lay = self._lay(ids)
            for l in range(l0 + 1, self.L):
                dsrc = self.N + torch.cat(dirty)
                if dsrc.numel() == 0:
                    break
                new_d = []
                for seg in self._chunks(ids[lay == l], self.rows):
                    rg, slot2, sc, sy, sx, hwl, _ = self._rows(seg)
                    taps = self._shift(self.conn[seg].long()[rg], sc, sy, sx, hwl)
                    r = torch.isin(taps, dsrc).any(1).nonzero().flatten()
                    if r.numel() == 0:
                        continue
                    left -= r.numel()
                    if left < 0:
                        ok = False
                        break
                    out = self.tt[seg][rg[r]].gather(
                        1, self._cells(taps[r], self.coef[seg][rg[r]], self.src, self.D).long())
                    rid = self.N + slot2[r]
                    df = pack_bits(out.to(torch.uint8)) ^ self.src[rid]
                    m2 = (df != 0).any(1)
                    if not int(m2.sum()):
                        continue
                    ob = unpack_bits(self.src[rid[m2]], self.D)
                    ds.index_add_(0, self._cls(slot2[r][m2]),
                                  (out[m2].float() - ob.float())
                                  * self.sgn[seg].float()[rg[r][m2], None])
                    self.src[rid[m2]] ^= df[m2]
                    applied.append((rid[m2], df[m2]))
                    new_d.append(slot2[r][m2])
                if not ok:
                    break
                dirty.append(torch.cat(new_d) if new_d else slot[:0])
        h1 = self._hinge(self.score + ds)
        if ok and h1 < self.hval + thr:
            self.score += ds
            self.hval = h1
            return True
        for rid, df in reversed(applied):
            self.src[rid] ^= df
        return False

    # -- lever 1: block-CD over all 2**K truth-table bits of each visited gate -------------
    @torch.no_grad()
    def tt_sweep(self, gid: torch.Tensor) -> int:
        rg, slot, sc, sy, sx, hwl, _ = self._rows(gid)
        cells = self._cells(self._shift(self.conn[gid].long()[rg], sc, sy, sx, hwl),
                            self.coef[gid][rg], self.src, self.D)
        cl = cells.long()
        ttg = self.tt[gid]
        o = ttg[rg].gather(1, cl)                                    # (R, D) current outputs
        _, tup, tdn = self._base_tables()
        rcls = self._cls(slot)
        sw = self.sgn[gid].float()[rg]                               # vote polarity per row
        neg = (self.sgn[gid] < 0)[rg, None]
        dlt = torch.where(o ^ neg, tdn[rcls], tup[rcls])             # delta if this row flips
        ben = torch.zeros(gid.numel() * self.TT, device=self.device)
        ben.scatter_add_(0, (cl + (rg * self.TT)[:, None]).flatten(), dlt.flatten())
        flip = ben.view(-1, self.TT) < -self.EPS                     # bits partition the rows;
        for _ in range(6):                                           # ben is the exact DIRECT
            n = int(flip.sum())                                      # delta -- the cascade is
            if n == 0:                                               # verified in _commit,
                return 0                                             # halving until it clears
            ttn = ttg ^ flip
            if self._commit(slot, o, ttn[rg].gather(1, cl), -self.EPS, sw, sw):
                self.tt[gid] = ttn
                return n
            flip[torch.rand(gid.numel(), device=self.device) < 0.5] = False
        return 0

    # -- lever 2: rewire one random connection to the best of n random candidates ----------
    @torch.no_grad()
    def rewire(self, gid: torch.Tensor, n_cand: int, radius: int = 0,
               local_frac: float = 0.0) -> int:
        g = gid.numel()
        rg, slot, sc, sy, sx, hwl, _ = self._rows(gid)
        hi = self.N + int(self.cum_slots[int(self._lay(gid[:1]))])   # sources below this layer
        flat = self._shift(self.conn[gid].long()[rg], sc, sy, sx, hwl)
        cells = self._cells(flat, self.coef[gid][rg], self.src, self.D)
        ttg = self.tt[gid]
        of = ttg[rg].gather(1, cells.long())
        _, tup, tdn = self._base_tables()
        rcls = self._cls(slot)
        sw = self.sgn[gid].float()[rg]                               # vote polarity per row:
        pos = (sw > 0)[:, None]                                      # output delta d moves the
        tu = torch.where(pos, tup[rcls], tdn[rcls])                  # vote by sgn*d, so swap
        td = torch.where(pos, tdn[rcls], tup[rcls])                  # the tables on neg rows
        k = torch.randint(self.K, (g,), device=self.device)
        ck = self.coef[gid, k][rg][:, None]                          # tap-k hash weight per row
        xk = unpack_bits(self.src[flat.gather(1, k[rg][:, None]).squeeze(1)], self.D)
        bas = (cells - ck * xk).remainder_(self.M)                   # hash with tap k removed
        cands = torch.randint(hi, (g, n_cand), device=self.device)
        for c in range(int(n_cand * local_frac) if radius else 0):  # local candidates: same
            cands[:, c] = self._near(self.conn[gid, k].long(), radius, 0.0, hi)  # neighborhood

        def outs(cand):                                              # cand (g,): per-row outputs
            fl = self._shift(cand[rg][:, None], sc, sy, sx, hwl)     # candidate coord, per copy
            cb = unpack_bits(self.src[fl[:, 0]], self.D)
            return ttg[rg].gather(1, torch.remainder(bas + ck * cb, self.M).long())

        gain = torch.empty((g, n_cand), device=self.device)
        for c in range(n_cand):
            d = outs(cands[:, c]).float() - of.float()
            gain[:, c] = torch.zeros(g, device=self.device).index_add_(
                0, rg, ((d > 0) * tu + (d < 0) * td).sum(1))
        best, bi = gain.min(1)
        acc = best < -self.EPS
        if not int(acc.sum()):
            return 0
        nw = outs(cands[torch.arange(g, device=self.device), bi])    # chosen candidate, all rows
        for _ in range(6):                                           # same cascaded joint check
            n = int(acc.sum())
            if n == 0:
                return 0
            rm = acc[rg]
            if self._commit(slot[rm], of[rm], nw[rm], -self.EPS, sw[rm], sw[rm]):
                self.conn[gid[acc], k[acc]] = cands[acc, bi[acc]].to(torch.int32)
                return n
            acc = acc & (torch.rand(g, device=self.device) >= 0.5)
        return 0

    # -- lever: COEF -- re-learn one tap's hash weight (the hash-gate-only lever) -----------
    @torch.no_grad()
    def coef_pass(self, gid: torch.Tensor, n_cand: int) -> int:
        """COEF move: re-learn one tap's integer hash weight c_k, the lever unique to the
        hash gate. Same shifted-gather trick as rewire -- bas = (h - c_k*x_k) mod M
        removes the tap from every (copy, sample) hash, then any candidate weight is
        scored exactly for the price of one add -- so one tap unpack block-scores the
        whole candidate set over all copies. c'=0 makes the tap inert (learned fan-in
        DOWN, the exact-neutral growth entry point of v3); a later nonzero revives it
        (fan-in UP); +-1 weights move toward the threshold family, 2c toward LUT-style
        place values. Fan-in and function family become things CD edits, not settings.
        Candidates: +-small walks from c, ternary corner draws, doubling, uniform."""
        g = gid.numel()
        rg, slot, sc, sy, sx, hwl, _ = self._rows(gid)
        flat = self._shift(self.conn[gid].long()[rg], sc, sy, sx, hwl)
        cells = self._cells(flat, self.coef[gid][rg], self.src, self.D)
        ttg = self.tt[gid]
        of = ttg[rg].gather(1, cells.long())
        _, tup, tdn = self._base_tables()
        rcls = self._cls(slot)
        sw = self.sgn[gid].float()[rg]
        pos = (sw > 0)[:, None]
        tu = torch.where(pos, tup[rcls], tdn[rcls])
        td = torch.where(pos, tdn[rcls], tup[rcls])
        k = torch.randint(self.K, (g,), device=self.device)
        ck = self.coef[gid, k][rg][:, None]                          # current weight per row
        xk = unpack_bits(self.src[flat.gather(1, k[rg][:, None]).squeeze(1)], self.D)
        bas = (cells - ck * xk).remainder_(self.M)                   # hash with tap k removed
        cur = self.coef[gid, k].long()[:, None]
        u = torch.rand(g, n_cand, device=self.device)
        pm = torch.randint(1, 4, (g, n_cand), device=self.device) \
            * (torch.randint(0, 2, (g, n_cand), device=self.device) * 2 - 1)
        cands = torch.where(u < 0.4, cur + pm,
                torch.where(u < 0.6, torch.randint(-1, 2, (g, n_cand), device=self.device),
                torch.where(u < 0.8, cur * 2,
                            torch.randint(self.M, (g, n_cand), device=self.device))))
        cands = cands.remainder_(self.M).to(torch.int16)

        def outs(cand):                                              # cand (g,): per-row outputs
            return ttg[rg].gather(
                1, torch.remainder(bas + cand[rg][:, None] * xk, self.M).long())

        gain = torch.empty((g, n_cand), device=self.device)
        for c in range(n_cand):
            d = outs(cands[:, c]).float() - of.float()
            gain[:, c] = torch.zeros(g, device=self.device).index_add_(
                0, rg, ((d > 0) * tu + (d < 0) * td).sum(1))
        best, bi = gain.min(1)
        acc = best < -self.EPS
        if not int(acc.sum()):
            return 0
        nw = outs(cands[torch.arange(g, device=self.device), bi])
        for _ in range(6):                                           # same cascaded joint check
            n = int(acc.sum())
            if n == 0:
                return 0
            rm = acc[rg]
            if self._commit(slot[rm], of[rm], nw[rm], -self.EPS, sw[rm], sw[rm]):
                self.coef[gid[acc], k[acc]] = cands[acc, bi[acc]]
                return n
            acc = acc & (torch.rand(g, device=self.device) >= 0.5)
        return 0

    # -- lever 3: share/unshare a gate along one dimension (evict / refill, exact accept) --
    @torch.no_grad()
    def share_move(self, g: int, dim: int, up: bool, neutral: bool = False) -> bool:
        dg = self.deg[g].long()
        l = int(self._lay(torch.tensor([g], device=self.device)))
        md = self.maxdeg_c[l] if dim == 0 else self.maxdeg_s[l]
        if up and (int(dg[dim]) >= md or 2 << int(dg.sum()) > self.max_copies):
            return False
        if not up and int(dg[dim]) == 0:
            return False
        dimsz = self.chs[l] if dim == 0 else self.hws[l]
        s0 = int(self.step[g, dim])
        if up:                                    # existing copies must stay bit-identical:
            if int(dg[dim]) == 0:                 # 2j*s1 == j*s0 forces s1 = s0/2
                s1 = dimsz >> 1
            elif s0 % 2 == 0:
                s1 = s0 >> 1
            else:
                return False                      # odd stride cannot interleave evenly
        else:
            s1 = (2 * s0) % dimsz                 # kept even copies: j*(2*s0) == (2j)*s0
        gid = torch.tensor([g], device=self.device)
        if up:
            self.deg[g, dim] += 1                                    # enumerate proposed copies
            self.step[g, dim] = s1
        rg, slot, sc, sy, sx, hwl, idx3 = self._rows(gid)
        if up:
            self.deg[g, dim] -= 1
            self.step[g, dim] = s0
        odd = idx3[:, dim] % 2 == 1                                  # up: the NEW copies;
        slot_o = slot[odd]                                           # down: the DROPPED copies
        o_g = self.tt[g][self._cells(self._shift(self.conn[gid].long()[rg[odd]], sc[odd],
                                                 sy[odd], sx[odd], hwl[odd]),
                                     self.coef[gid][rg[odd]],
                                     self.src, self.D).long()]       # g's outputs on those slots
        if up:                                                       # evict the current owners
            ev = torch.unique(self.owner[slot_o].long())
            erg, eslot, esc, esy, esx, ehw, _ = self._rows(ev)
            eout = self.tt[ev][erg].gather(
                1, self._cells(self._shift(self.conn[ev].long()[erg], esc, esy, esx, ehw),
                               self.coef[ev][erg], self.src, self.D).long())
            self._smask[slot_o] = True
            tk = self._smask[eslot]                                  # eslot rows taken by g
            self._smask[slot_o] = False
            freed = eslot[~tk]                                       # evicted slots not taken
            self._sidx[slot_o] = torch.arange(slot_o.numel(), device=self.device)
            nw = eout.clone()
            nw[tk] = o_g[self._sidx[eslot[tk]]]
            slots, old = eslot, eout
            wo = self.sgn[ev].float()[erg]                           # old owners' polarity
            wn = torch.where(tk, torch.full_like(wo, float(self.sgn[g])),
                             torch.ones_like(wo))                    # g's / fresh (+1)
        else:
            freed, slots, old = slot_o, slot_o, o_g
            wo = torch.full((slot_o.numel(),), float(self.sgn[g]), device=self.device)
            wn = torch.ones(slot_o.numel(), device=self.device)
        f = freed.numel()
        _, _, fy, fx, _ = self._cyx(freed)
        conn_f = self._rand_conn(l, fy, fx)
        tt_f, coef_f = self._rand_fn(f)
        fout = tt_f.gather(1, self._cells(conn_f.long(), coef_f, self.src, self.D).long()) \
            if f else o_g[:0]                                        # fresh unshared randoms
        if up:
            nw[~tk] = fout
        else:
            nw = fout
        if not self._commit(slots, old, nw, 1e-6 if neutral else -self.EPS, wo, wn):
            return False
        self.deg[g, dim] += 1 if up else -1
        self.step[g, dim] = s1
        if up:
            self.alive[ev] = False
            self.owner[slot_o] = g
        if f:
            ids = (~self.alive).nonzero().flatten()[:f]
            self.base[ids] = freed.to(torch.int32)
            self.conn[ids], self.tt[ids], self.coef[ids] = conn_f, tt_f, coef_f
            self.deg[ids] = 0
            self.step[ids] = 0
            self.sgn[ids] = 1
            self.alive[ids] = True
            self.owner[freed] = ids.to(torch.int32)
        return True

    # -- lever 4: RANDOM SEARCH -- joint random mutation packages -------------------------
    @torch.no_grad()
    def rs_pass(self, gid: torch.Tensor, p_rew: float, n_bits: int, radius: int,
                p_global: float, temp: float, neutral: bool) -> int:
        """RANDOM SEARCH pass: each given gate gets ONE random LOCAL mutation -- with prob
        p_rew a connection is re-tapped in its spatial neighborhood (+-radius; p_global
        chance of a long-range jump), else a BURST of n_bits tt bits flips jointly (a
        coordinated multi-bit change can reach functions no single improving flip can,
        i.e. features currently uncorrelated with the error signal). Each gate's mutation
        is accepted INDEPENDENTLY on its own exact hinge delta: improving always; equal
        ('neutral', drift along flat regions); or up to `temp` worse (bounded uphill).
        Per-gate acceptance keeps RS alive on a polished net. At depth 0 the per-gate
        delta IS exact; with depth the accepted set is verified as a joint package on the
        exact cascaded hinge (halved until it clears)."""
        rg, slot, sc, sy, sx, hwl, _ = self._rows(gid)
        g = gid.numel()
        gr = torch.arange(g, device=self.device)
        hi = self.N + int(self.cum_slots[int(self._lay(gid[:1]))])
        conn_n = self.conn[gid].clone()
        tt_n = self.tt[gid].clone()
        rw = torch.rand(g, device=self.device) < p_rew               # per-gate axis choice
        if int(rw.sum()):
            ks = torch.randint(self.K, (g,), device=self.device)
            near = self._near(conn_n[gr, ks].long(), radius, p_global, hi).to(torch.int32)
            conn_n[gr[rw], ks[rw]] = near[rw]
        nm = ~rw
        if int(nm.sum()):
            for _ in range(max(1, n_bits)):
                kb = torch.randint(self.TT, (g,), device=self.device)
                tt_n[gr[nm], kb[nm]] ^= True
        old = self.tt[gid][rg].gather(
            1, self._cells(self._shift(self.conn[gid].long()[rg], sc, sy, sx, hwl),
                           self.coef[gid][rg], self.src, self.D).long())
        nw = tt_n[rg].gather(
            1, self._cells(self._shift(conn_n.long()[rg], sc, sy, sx, hwl),
                           self.coef[gid][rg], self.src, self.D).long())
        d = nw.float() - old.float()
        _, tup, tdn = self._base_tables()
        rcls = self._cls(slot)
        sw = self.sgn[gid].float()[rg]
        pos = (sw > 0)[:, None]
        per = ((d > 0) * torch.where(pos, tup[rcls], tdn[rcls])      # exact DIRECT per-row
               + (d < 0) * torch.where(pos, tdn[rcls], tup[rcls])).sum(1)  # delta, polarity-aware
        gd = torch.zeros(g, device=self.device).index_add_(0, rg, per)
        thr = temp if temp > 0 else (1e-6 if neutral else -self.EPS)
        acc = gd <= thr
        if self.L == 1:                                              # depth 0: per-gate exact
            n = int(acc.sum())
            if n == 0:
                return 0
            rm = acc[rg]
            ds = torch.zeros_like(self.score)
            ds.index_add_(0, rcls[rm], d[rm] * sw[rm][:, None])
            self.conn[gid[acc]] = conn_n[acc]
            self.tt[gid[acc]] = tt_n[acc]
            self.score += ds
            self.hval = self._hinge(self.score)  # per-gate deltas are not jointly additive
            return n
        for _ in range(6):                                           # depth: cascaded package
            n = int(acc.sum())
            if n == 0:
                return 0
            rm = acc[rg]
            if self._commit(slot[rm], old[rm], nw[rm], thr, sw[rm], sw[rm]):
                self.conn[gid[acc]] = conn_n[acc]
                self.tt[gid[acc]] = tt_n[acc]
                return n
            acc = acc & (torch.rand(g, device=self.device) >= 0.5)
        return 0

    # -- lever 5: SPLIT -- unshare by cloning (net2net-style), exactly output-neutral ------
    @torch.no_grad()
    def split_move(self, g: int, dim: int) -> bool:
        """Split a shared gate along one dimension: even copies keep gate g, odd copies go
        to an identical clone. Outputs are bit-for-bit unchanged (hinge-exactly-neutral,
        no accept test, NO CASCADE), but the two halves can now differentiate under CD/RS
        -- capacity grows precisely where position-specific features later pay, without
        the random-refill penalty that makes plain share-down hard to accept. (Uniform
        per-grid step shifts make this exact along ALL dims, channels included.)"""
        dg = self.deg[g]
        if int(dg[dim]) == 0:
            return False
        free = (~self.alive).nonzero().flatten()
        if free.numel() == 0:
            return False
        gid = torch.tensor([g], device=self.device)
        _, slot, _, _, _, _, idx3 = self._rows(gid)
        slot_o = slot[idx3[:, dim] % 2 == 1]
        nid = free[0]
        l = int(self._lay(gid))
        cl, hwl = self.chs[l], self.hws[l]
        b = int(self.base[g]) - int(self.cum_slots[l])
        c, y, x = b // (hwl * hwl), (b // hwl) % hwl, b % hwl        # layer-local coords
        dimsz = cl if dim == 0 else hwl
        outst = dimsz >> int(dg[dim])           # output tiling stride along the split dim
        st = int(self.step[g, dim])             # learned INPUT stride along it
        c = (c + outst) % cl if dim == 0 else c
        y = (y + outst) % hwl if dim == 1 else y
        x = (x + outst) % hwl if dim == 2 else x
        self.base[nid] = int(self.cum_slots[l]) + (c * hwl + y) * hwl + x
        # the odd copies' +st input shift came from their copy index; bake it into the
        # clone's connections so every copy's total input shift is bit-for-bit unchanged
        z = torch.zeros(1, dtype=torch.long, device=self.device)
        s = torch.full((1,), st, dtype=torch.long, device=self.device)
        hwt = torch.full((1,), hwl, dtype=torch.long, device=self.device)
        self.conn[nid] = self._shift(self.conn[g][None].long(),
                                     s if dim == 0 else z,
                                     s if dim == 1 else z,
                                     s if dim == 2 else z, hwt)[0].to(torch.int32)
        self.tt[nid] = self.tt[g]
        self.coef[nid] = self.coef[g]
        self.sgn[nid] = self.sgn[g]
        self.deg[g, dim] -= 1
        self.step[g, dim] = (2 * st) % dimsz    # kept/clone copies: j*(2st) == (2j)*st
        self.deg[nid] = self.deg[g]
        self.step[nid] = self.step[g]
        self.alive[nid] = True
        self.owner[slot_o] = nid.to(torch.int32)
        return True

    # -- lever 6: STEP -- re-stride the input side of a shared dimension -------------------
    @torch.no_grad()
    def step_pass(self, gid: torch.Tensor) -> int:
        """STEP move: mutate the learned input stride of one shared dimension per gate.
        The output tiling never moves; only what the copies READ shifts. step = dim/n is
        plain conv striding, smaller overlaps receptive fields, larger dilates, 0 ties all
        copies to identical inputs (classic exact weight sharing). Proposals mix local
        edits (halve, double, +-1) with uniform re-draws; the package is verified on the
        exact (cascaded) hinge like every other move."""
        sh = self.deg[gid].long().sum(1) > 0
        gid = gid[sh]
        g = gid.numel()
        if g == 0:
            return 0
        rg, slot, sc, sy, sx, hwl, _ = self._rows(gid)
        old = self.tt[gid][rg].gather(
            1, self._cells(self._shift(self.conn[gid].long()[rg], sc, sy, sx, hwl),
                           self.coef[gid][rg], self.src, self.D).long())
        d3 = torch.multinomial((self.deg[gid] > 0).float(), 1).flatten()
        lay = self._lay(gid)
        dimsz = torch.where(d3 == 0, self.chs_t[lay], self.hws_t[lay])
        cur = self.step[gid, d3].long()
        u = torch.rand(g, device=self.device)
        pm = torch.randint(0, 2, (g,), device=self.device) * 2 - 1
        cand = torch.where(u < 0.25, cur // 2,
               torch.where(u < 0.5, (cur * 2) % dimsz,
               torch.where(u < 0.75, (cur + pm) % dimsz,
                           (torch.rand(g, device=self.device) * dimsz).long())))
        self.step[gid, d3] = cand.to(torch.int16)
        rg2, _, sc2, sy2, sx2, hwl2, _ = self._rows(gid)             # same rows, new shifts
        nw = self.tt[gid][rg2].gather(
            1, self._cells(self._shift(self.conn[gid].long()[rg2], sc2, sy2, sx2, hwl2),
                           self.coef[gid][rg2], self.src, self.D).long())
        self.step[gid, d3] = cur.to(torch.int16)                     # decide, then write
        sw = self.sgn[gid].float()[rg]
        acc = torch.ones(g, dtype=torch.bool, device=self.device)
        for _ in range(6):
            n = int(acc.sum())
            if n == 0:
                return 0
            rm = acc[rg]
            if self._commit(slot[rm], old[rm], nw[rm], -self.EPS, sw[rm], sw[rm]):
                self.step[gid[acc], d3[acc]] = cand[acc].to(torch.int16)
                return n
            acc = acc & (torch.rand(g, device=self.device) >= 0.5)
        return 0

    # -- lever 7: SIGN -- flip a gate's vote polarity (Dale's law; cascade-free) -----------
    @torch.no_grad()
    def sign_pass(self, gid: torch.Tensor) -> int:
        """SIGN move: flip vote polarity, so a gate's slots count AGAINST their class
        instead of for it (features as negative evidence -- inhibitory populations).
        Outputs and all readers are untouched, so there is NO cascade at any depth: the
        cheapest lever in the system. Proposal = +-1-table estimate applied twice per unit
        vote; the package is verified on the exact hinge with halving."""
        rg, slot, sc, sy, sx, hwl, _ = self._rows(gid)
        out = self.tt[gid][rg].gather(
            1, self._cells(self._shift(self.conn[gid].long()[rg], sc, sy, sx, hwl),
                           self.coef[gid][rg], self.src, self.D).long())
        _, tup, tdn = self._base_tables()
        rcls = self._cls(slot)
        sw = self.sgn[gid].float()[rg]
        pos = (sw > 0)[:, None]
        est = torch.zeros(gid.numel(), device=self.device).index_add_(
            0, rg, (out.float() * 2 * torch.where(pos, tdn[rcls], tup[rcls])).sum(1))
        flip = est < -self.EPS
        for _ in range(6):
            n = int(flip.sum())
            if n == 0:
                return 0
            rm = flip[rg]
            if self._commit(slot[rm], out[rm], out[rm], -self.EPS, sw[rm], -sw[rm]):
                self.sgn[gid[flip]] = -self.sgn[gid[flip]]
                return n
            flip = flip & (torch.rand(gid.numel(), device=self.device) >= 0.5)
        return 0

    # -- lever: OUTPUT-CLASS -- relearn which class each gate votes for (learned readout) ---
    @torch.no_grad()
    def cls_pass(self, gid: torch.Tensor) -> int:
        """CLASS move: relearn each gate's OUTPUT connection -- which of the CLS classes its
        vote lands on -- instead of the fixed channel%CLS assignment. A great feature stuck in
        the 'wrong' class can be re-credited to the right one, and the search can reallocate
        class quota toward hard classes. Cascade-FREE (stored outputs untouched, only the
        head's class attribution changes, like sign_pass). Proposal from the +-1 tables: per
        gate, estimate the exact-first-order hinge change of moving its whole vote off its
        current class onto each candidate, take the best; verify the package on the exact
        hinge with random halving (the joint effect is nonlinear)."""
        rg, slot, sc, sy, sx, hwl, _ = self._rows(gid)
        out = self.tt[gid][rg].gather(
            1, self._cells(self._shift(self.conn[gid].long()[rg], sc, sy, sx, hwl),
                           self.coef[gid][rg], self.src, self.D).long())
        outf = out.float()
        _, tup, tdn = self._base_tables()
        rcls = self._cls(slot)                                       # current class per copy
        sw = self.sgn[gid].float()[rg]
        pos = (sw > 0)[:, None]
        # removing the current vote: +w is a -1 change to its class (tdn), -w is +1 (tup)
        rem = (outf * torch.where(pos, tdn[rcls], tup[rcls])).sum(1)             # (R,)
        # adding it to candidate class c': +w is +1 (tup[c']), -w is -1 (tdn[c'])
        add = torch.where(pos, outf @ tup.t(), outf @ tdn.t())                   # (R, CLS)
        G = gid.numel()
        gest = torch.zeros(G, CLS, device=self.device).index_add_(0, rg, add)
        gest += torch.zeros(G, device=self.device).index_add_(0, rg, rem)[:, None]
        cur = self._cls(self.base[gid].long())                      # gate's base-slot class
        gest.scatter_(1, cur[:, None], float("inf"))                # never re-pick current
        best = gest.argmin(1)
        mv = (gest.gather(1, best[:, None]).squeeze(1) < -self.EPS)
        cnew_row = best[rg]
        for _ in range(6):
            n = int(mv.sum())
            if n == 0:
                return 0
            rm = mv[rg]
            v = sw[rm][:, None] * outf[rm]
            ds = torch.zeros_like(self.score)
            ds.index_add_(0, rcls[rm], -v)
            ds.index_add_(0, cnew_row[rm], v)
            h1 = self._hinge(self.score + ds)
            if h1 < self.hval - self.EPS:
                self.score += ds
                self.hval = h1
                self.ocls[slot[rm]] = cnew_row[rm].to(torch.int8)
                return n
            mv = mv & (torch.rand(G, device=self.device) >= 0.5)
        return 0

    # -- lever 8: REBUILD -- prune-and-regrow harmful gates (SET-style, exact accepts) -----
    @torch.no_grad()
    def rebuild_pass(self, gid: torch.Tensor) -> int:
        """PRUNE-AND-REGROW with counterfactual credit (difference rewards from multiagent
        RL x sparse evolutionary training): estimate each gate's marginal vote value from
        the exact +-1 tables; a gate whose REMOVAL would lower the hinge is deadwood-or-
        worse, so propose replacing its function and wiring with a fresh random gate (same
        locality/residual mix as init, sharing geometry kept), accepted as an exact
        (cascaded) package. Turns measured harm into fresh exploration capacity."""
        rg, slot, sc, sy, sx, hwl, _ = self._rows(gid)
        old = self.tt[gid][rg].gather(
            1, self._cells(self._shift(self.conn[gid].long()[rg], sc, sy, sx, hwl),
                           self.coef[gid][rg], self.src, self.D).long())
        _, tup, tdn = self._base_tables()
        rcls = self._cls(slot)
        sw = self.sgn[gid].float()[rg]
        pos = (sw > 0)[:, None]
        est = torch.zeros(gid.numel(), device=self.device).index_add_(
            0, rg, (old.float() * torch.where(pos, tdn[rcls], tup[rcls])).sum(1))
        rb = est < -self.EPS                                         # removal would help
        if not int(rb.sum()):
            return 0
        lay, _, by, bx, _ = self._cyx(self.base[gid].long())
        conn_f = self.conn[gid].clone()
        for l in torch.unique(lay).tolist():                         # fresh taps, per layer
            m = lay == l
            conn_f[m] = self._rand_conn(int(l), by[m], bx[m])
        tt_f, coef_f = self._rand_fn(gid.numel())
        nw = tt_f[rg].gather(
            1, self._cells(self._shift(conn_f.long()[rg], sc, sy, sx, hwl),
                           coef_f[rg], self.src, self.D).long())
        for _ in range(6):
            n = int(rb.sum())
            if n == 0:
                return 0
            rm = rb[rg]
            if self._commit(slot[rm], old[rm], nw[rm], -self.EPS, sw[rm],
                            torch.ones_like(sw[rm])):
                self.conn[gid[rb]] = conn_f[rb]
                self.tt[gid[rb]] = tt_f[rb]
                self.coef[gid[rb]] = coef_f[rb]
                self.sgn[gid[rb]] = 1
                return n
            rb = rb & (torch.rand(gid.numel(), device=self.device) >= 0.5)
        return 0

    # -- debug: verify the incremental state against a from-scratch pass -------------------
    def check(self) -> float:
        """Max deviation between incremental state and a from-scratch pass: the score AND,
        with depth, every stored output row. Must be 0.0."""
        if self.L == 1:
            return (self.forward(self.src, self.D, self.rows) - self.score).abs().max().item()
        src2 = self.src.clone()
        sc = self.forward(src2, self.D, self.rows)
        return max((sc - self.score).abs().max().item(),
                   float((src2 != self.src).sum().item()))

    def copy_stats(self) -> tuple[float, int, float]:
        n = 1 << self.deg[self.alive].long().sum(1)
        return float(n.float().mean()), int(n.max()), 100.0 * float((n > 1).float().mean())


# ==========================================================================================
def augment(x: torch.Tensor, crop: int = 4, jitter: float = 0.0, cut: int = 0) -> torch.Tensor:
    """Standard CIFAR augmentation on (D,3,32,32) [0,1] images: random horizontal flip,
    random crop from `crop`-pixel replicate padding, optional brightness/contrast jitter,
    optional cut x cut cutout (gray fill). Re-rolled every round, this is stochastic
    augmentation like SGD sees: the hinge can never permanently reach zero, so optimization
    pressure never dies. (Cutout was removed while the net underfit; re-added once
    init-deg 0,0,0 pushed us to a 24-pt train/val gap -- overfit regime, re-testing.)"""
    d = x.shape[0]
    fl = torch.rand(d) < 0.5
    x = torch.where(fl[:, None, None, None], x.flip(-1), x)
    if jitter:
        b = (torch.rand(d, 1, 1, 1) - 0.5) * 0.3 * jitter            # brightness shift
        c = 1.0 + (torch.rand(d, 1, 1, 1) - 0.5) * 0.4 * jitter      # contrast scale
        m = x.mean((1, 2, 3), keepdim=True)
        x = ((x - m) * c + m + b).clamp_(0, 1)
    if cut:
        oy = torch.randint(0, 33 - cut, (d, 1))
        ox = torch.randint(0, 33 - cut, (d, 1))
        rng = torch.arange(32)[None, :]
        m = ((rng >= oy) & (rng < oy + cut))[:, :, None] \
            & ((rng >= ox) & (rng < ox + cut))[:, None, :]
        x = torch.where(m[:, None, :, :], torch.full_like(x, 0.5), x)
    if crop:
        p = torch.nn.functional.pad(x, (crop,) * 4, mode="replicate").permute(0, 2, 3, 1)
        oy = torch.randint(0, 2 * crop + 1, (d,))
        ox = torch.randint(0, 2 * crop + 1, (d,))
        b = torch.arange(d)[:, None, None]
        ys = (oy[:, None] + torch.arange(32))[:, :, None]            # (d, 32, 1)
        xs = (ox[:, None] + torch.arange(32))[:, None, :]            # (d, 1, 32)
        x = p[b, ys, xs].permute(0, 3, 1, 2)
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/cifar-10-batches-py"))
    p.add_argument("--download", action="store_true")
    p.add_argument("--train-size", type=int, default=0, help="train+val pool (0 = full 50k)")
    p.add_argument("--num-bits", type=int, default=5)
    p.add_argument("--channels", type=str, default="30720",
                   help="window channels per layer, comma-separated: '30720' = the flat "
                        "depth-0 model (31.5M slots); '640,320,...' = stacked layers, "
                        "gates tap any lower layer or the input")
    p.add_argument("--spatial", type=str, default="32",
                   help="spatial grid size per layer (one value = all layers): e.g. "
                        "'32,32,16,16,8,8,4,4' is a CNN-style pooling pyramid -- coarser "
                        "layers cost fewer slots and see wider context")
    p.add_argument("--fan-in", type=int, default=4)
    p.add_argument("--gate", choices=["lut", "hash", "ternary"], default="lut",
                   help="fresh-gate corner of the hash-gate space T[(sum c_k x_k) mod M]: "
                        "'lut' = c_k=2**k + random tables (the classic LUT gate, bit-exact "
                        "with the old executor); 'hash' = c uniform in [1,M) + random "
                        "tables (full-table occupancy at any K/M, the only corner for "
                        "K>8); 'ternary' = c in {-1,0,1} + threshold-step tables "
                        "(BitNet-style -- measured badly behind in hg1: only 2K+1 of M "
                        "cells reachable at init). Init only -- cd-cf moves every gate "
                        "freely in the space either way")
    p.add_argument("--tsize", type=int, default=0,
                   help="table size M per gate (0 = 2**fan_in, the classic coupling). "
                        "Decoupling M from K breaks the K<=8 LUT barrier: eval O(K), "
                        "storage O(M) -- e.g. --fan-in 16 --tsize 64 --gate hash gives "
                        "16-tap gates with 64-bit tables (impossible as a 2**16 LUT)")
    p.add_argument("--n-cand", type=int, default=8, help="candidate inputs per rewire visit")
    p.add_argument("--rewire-frac", type=float, default=0.25,
                   help="fraction of gates rewire-visited per round")
    p.add_argument("--share-moves", type=int, default=256, help="share attempts per round")
    p.add_argument("--rs-frac", type=float, default=0.25,
                   help="fraction of gates getting one random mutation per round")
    p.add_argument("--rs-bits", type=int, default=3,
                   help="tt bits flipped JOINTLY per RS tt visit (local multi-bit burst)")
    p.add_argument("--rs-radius", type=int, default=4,
                   help="spatial neighborhood (pixels) for local connection moves")
    p.add_argument("--rs-global-p", type=float, default=0.1,
                   help="prob an RS connection move jumps anywhere instead of locally")
    p.add_argument("--local-frac", type=float, default=0.5,
                   help="fraction of CD rewire candidates drawn from the neighborhood")
    p.add_argument("--rs-temp", type=float, default=0.0,
                   help=">0: also accept mutations up to this much hinge WORSE (annealing)")
    p.add_argument("--rs-shares", type=int, default=64,
                   help="neutral-accept random share moves per round (sharing axis of RS)")
    p.add_argument("--splits", type=int, default=256,
                   help="budgeted split attempts per round (unshare-by-cloning, free/neutral)")
    p.add_argument("--split-batch", type=int, default=512,
                   help="split attempts per 'sp' work unit -- splits are output-neutral "
                        "(free capacity) so the bandit can't price their deferred value; "
                        "batching compensates. Wave-1: 512 beat 256 (val 43.0 vs 40.1)")
    p.add_argument("--rs-neutral", type=int, default=1,
                   help="1: accept hinge-neutral mutations (drift along flat regions)")
    p.add_argument("--heat", type=int, default=1,
                   help="1: prioritized gate visiting (RL prioritized-replay port): chunks "
                        "sample gates by an EMA of recent hinge yield, floor keeps cold "
                        "gates covered; 0: uniform shuffles")
    p.add_argument("--explore", type=str, default="1",
                   help="exploration multiplier scaling the whole RS side (rs-frac, "
                        "rs-shares, rs-temp): a constant 'e', or 'start:end:rounds' for a "
                        "linear schedule (e.g. 1:0.3:300)")
    p.add_argument("--max-copies", type=int, default=1024, help="cap on copies per gate")
    p.add_argument("--pass-rows", type=int, default=2048, help="window slots per batched call")
    p.add_argument("--casc-cap", type=int, default=65536,
                   help="max recomputed reader rows per cascade before the move is rejected")
    p.add_argument("--rounds", type=int, default=10 ** 9)
    p.add_argument("--work-frac", type=float, default=1.0,
                   help="scale the per-round work budget: <1 = shorter rounds, so more "
                        "frequent logging/eval points (a denser learning curve, faster)")
    p.add_argument("--val-every", type=int, default=1, help="rounds between val/test evals")
    p.add_argument("--aug", choices=["none", "full"], default="none",
                   help="full: flip+crop+jitter, re-rolled every --aug-every rounds")
    p.add_argument("--aug-crop", type=int, default=4)
    p.add_argument("--aug-jitter", type=float, default=0.0)
    p.add_argument("--aug-cut", type=int, default=0)
    p.add_argument("--aug-every", type=int, default=1)
    p.add_argument("--init-deg", type=str, default="0,0,0",
                   help="initial log2 sharing degrees per dim c,h,w. Waves 1-2 measured (deep "
                        "net): LESS init sharing monotonically wins -- 0,0,0 (fully "
                        "unshared) val 50.1 > 0,1,1 44.2 > 0,2,2 40.1 > 0,3,3 37.7. Hand "
                        "the search NO conv prior; it learns sharing via share-up where it "
                        "pays. (Overfits from here -> regularization is the next lever.)")
    p.add_argument("--init-loc", type=int, default=2,
                   help="tap locality radius at init/refill (0 = uniform wiring): taps "
                        "start within +-R pixels of the gate's own position -- the CNN "
                        "locality prior, undoable by rewiring. A/B measured: loc=2+res=.5 "
                        "beat uniform 36.5 vs 31.8 val at equal budget (uniform trains "
                        "higher, generalizes worse)")
    p.add_argument("--init-res", type=float, default=0.5,
                   help="fraction of fresh truth tables initialized as pass-through of "
                        "tap 0 (residual init: signal flows through depth from round 0)")
    p.add_argument("--ckpt", type=Path, default=None, help="save the model here every eval")
    p.add_argument("--resume", action="store_true", help="load --ckpt before training")
    p.add_argument("--check", action="store_true", help="verify incremental scores each round")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    dev = args.device
    print(f"args={vars(args)}", flush=True)

    tx, ty, ex, ey = load_cifar10(args.data_dir, args.download)
    if args.train_size > 0:
        tx, ty = tx[: args.train_size], ty[: args.train_size]
    nv = max(1, round(len(tx) * 0.1))
    vx, vy, px, py = tx[-nv:], ty[-nv:], tx[:-nv], ty[:-nv]
    enc = Thermometer(num_bits=args.num_bits).fit(px[:2000])

    def encode(images):
        return enc(images).flatten(1).t().contiguous().to(torch.uint8)

    Xva, Xte = encode(vx), encode(ex)
    chs = [int(v) for v in args.channels.split(",")]
    hws = [int(v) for v in args.spatial.split(",")]
    if len(hws) == 1:
        hws = hws * len(chs)
    win = Win(3 * args.num_bits, 32, chs, hws, args.fan_in, args.max_copies, dev,
              init_deg=tuple(int(v) for v in args.init_deg.split(",")),
              init_loc=args.init_loc, init_res=args.init_res, gate=args.gate,
              tsize=args.tsize)
    win.cap = args.casc_cap
    spath = args.ckpt.with_suffix(".jsonl") if args.ckpt else None
    r0 = 0
    if args.resume and args.ckpt and args.ckpt.exists():
        ck = torch.load(args.ckpt, map_location=dev)
        for k in ("base", "conn", "tt", "coef", "deg", "alive", "owner", "step", "sgn",
                  "ocls"):                       # old ckpts lack coef -> keep the 2**k init
            if k in ck:
                getattr(win, k).copy_(ck[k])
        r0 = ck["round"]
        print(f"resumed {args.ckpt} (round {r0})", flush=True)
    elif spath:
        spath.write_text("")                                         # fresh run: fresh stats

    def reroll():
        """The round's training view: the full train set, freshly augmented. Every accept
        is exact on all of it -- per-round re-rolls are the stochasticity. (A per-round
        random-batch mode was tried and measured strictly worse: accepts overfit the
        batch and val crawled; exact full-train accepts won.)"""
        xb = augment(px, args.aug_crop, args.aug_jitter, args.aug_cut) if args.aug == "full" else px
        win.set_train(encode(xb), py, args.pass_rows)

    t0 = time.time()
    reroll()
    print(f"N={win.N} D={win.D} slots={win.S} layers={win.L} ({args.channels} @ "
          f"{','.join(map(str, hws))}) K={win.K} tau={win.tau:.0f}  "
          f"initial scores in {time.time()-t0:.0f}s  train={win.train_acc():.2f}", flush=True)
    if spath and not (args.resume and r0):                        # round 0 = random baseline
        v0 = win.evaluate(Xva, vy, args.pass_rows)
        te0 = win.evaluate(Xte, ey, args.pass_rows)
        with open(spath, "a") as f:
            f.write(json.dumps({"round": 0, "train": round(win.train_acc(), 2),
                                "val": round(v0, 2), "test": round(te0, 2),
                                "hinge": round(win.hval / win.D, 4), "ttbits": 0,
                                "rewires": 0, "shares": 0, "rs": 0, "clsmoves": 0,
                                "rebuilds": 0, "gates": int(win.alive.sum())}) + "\n")

    print(f"{'round':>5} | {'ttbits':>9} | {'rewires':>8} | {'shares':>6} | {'rs':>5} | "
          f"{'cp':>5} | {'train':>6} | {'hinge':>6} | {'val':>6} | {'test':>6} | {'min':>6}",
          flush=True)
    t0 = last_log = time.time()

    def prog(tag, rnd, done, tot, n):
        nonlocal last_log
        if time.time() - last_log > 60:
            last_log = time.time()
            print(f"    r{rnd} {tag} {done}/{tot} acc={n} train={win.train_acc():.2f}", flush=True)

    # adaptive operator selection: q[arm] is an EMA of the arm's measured reward (exact
    # hinge decrease per second); persists across rounds (bandit memory). With depth each
    # chunk operator becomes one arm PER LAYER, so the bandit also learns which layers are
    # currently worth their (cascade) cost.
    CHOPS = ("cd-tt", "cd-cn", "cd-cf", "cd-st", "cd-sg", "cd-cl", "cd-rb", "rs-cn", "rs-tt")
    OPS = tuple(f"{o}@{l}" for o in CHOPS for l in range(win.L)) + ("cd-sh", "rs-sh", "sp")
    opq = dict.fromkeys(OPS, 0.0)
    exs = [float(v) for v in args.explore.split(":")]

    def rows_for(l):                                                 # smaller packages deeper
        return max(256, args.pass_rows >> (win.L - 1 - l))           # down: sparser cascades

    heat = torch.zeros(win.S, device=dev)    # per-gate EMA of recent hinge yield: visiting
                                             # priority (prioritized-replay port from RL)
    def gate_stream(l):                      # recycling chunks, hot gates first
        while True:
            gs = win.alive.nonzero().flatten()
            gs = gs[win._lay(gs) == l]
            if args.heat:
                w = heat[gs] + heat[gs].mean() + 1e-6                # floor: cold gates too
                key = torch.rand(gs.numel(), device=dev).pow(1.0 / w)
                gs = gs[key.argsort(descending=True)]                # weighted order, no repl.
            else:
                gs = gs[torch.randperm(gs.numel(), device=dev)]
            for s in win._chunks(gs, rows_for(l)):
                yield s

    streams = {f"{o}@{l}": gate_stream(l) for o in CHOPS for l in range(win.L)}

    for rnd in range(r0 + 1, args.rounds + 1):
        if args.aug == "full" and rnd > r0 + 1 and (rnd - r0 - 1) % args.aug_every == 0:
            reroll()                             # (the round-1 view comes from init above)
        # RL-style adaptive operator sampling: every work unit SAMPLES an arm from
        #   p = e/K + (1-e) * q / sum(q)
        # (epsilon-greedy x probability-matching): e=1 -> uniform exploration over
        # arms, e->0 -> allocation proportional to what currently pays. q is the
        # per-arm EMA of exact hinge decrease per second, so cheap arms and
        # effective arms both earn share; the e/K floor keeps every arm alive
        # (neutral moves earn ~0 instant reward but are insurance, funded by the floor).
        ids = win.alive.nonzero().flatten()

        def share_try(neutral):
            g = int(ids[torch.randint(ids.numel(), (1,), device=dev)])
            if not bool(win.alive[g]):
                return 0
            return int(win.share_move(g, int(torch.randint(3, (1,))),
                                      bool(torch.randint(2, (1,))), neutral=neutral))

        e = exs[0] if len(exs) == 1 else \
            exs[0] + (exs[1] - exs[0]) * min(1.0, (rnd - 1) / max(1.0, exs[2]))
        temp = args.rs_temp * e
        nch = max(1, -(-win.S // args.pass_rows))                    # chunks per full sweep
                                                                     # (rows == slots: exact tiling)
        budget = (int(nch * (1 + args.rewire_frac + args.rs_frac * e))
                  + args.share_moves + int((args.rs_shares + args.splits) * e))
        budget = max(1, int(budget * args.work_frac))                # denser logging if <1
        cnt = dict.fromkeys(OPS, 0)
        nun = dict.fromkeys(OPS, 0)
        for u in range(budget):
            # (Thompson-sampling arm choice was A/B'd and refuted hard: it collapsed onto
            # one cheap-reward arm and starved tt/rewire/splits entirely -- the epsilon
            # floor is what keeps this non-stationary bandit honest.)
            qs = torch.tensor([max(0.0, opq[o]) for o in OPS])
            p = (e / len(OPS) + (1 - e) * qs / qs.sum()) if qs.sum() > 0 else \
                torch.full((len(OPS),), 1.0 / len(OPS))
            op = OPS[int(torch.multinomial(p, 1))]
            h0 = win.hval
            tu = time.time()
            if op == "sp":
                for _ in range(args.split_batch):                    # free moves: batch them
                    g = int(ids[torch.randint(ids.numel(), (1,), device=dev)])
                    if bool(win.alive[g]):
                        cnt[op] += int(win.split_move(g, int(torch.randint(3, (1,)))))
            elif op in ("cd-sh", "rs-sh"):
                cnt[op] += share_try(op == "rs-sh" and bool(args.rs_neutral))
            else:
                o, l = op.split("@")
                seg = next(streams[op])
                seg = seg[win.alive[seg]]                            # shares may have killed
                if seg.numel():
                    seg = seg[win._lay(seg) == int(l)]               # refills can change layer
                if seg.numel() == 0:
                    continue
                if o == "cd-tt":
                    cnt[op] += win.tt_sweep(seg)
                elif o == "cd-cn":
                    cnt[op] += win.rewire(seg, args.n_cand, args.rs_radius, args.local_frac)
                elif o == "cd-cf":
                    cnt[op] += win.coef_pass(seg, args.n_cand)
                elif o == "cd-st":
                    cnt[op] += win.step_pass(seg)
                elif o == "cd-sg":
                    cnt[op] += win.sign_pass(seg)
                elif o == "cd-cl":
                    cnt[op] += win.cls_pass(seg)
                elif o == "cd-rb":
                    cnt[op] += win.rebuild_pass(seg)
                else:
                    cnt[op] += win.rs_pass(seg, 1.0 if o == "rs-cn" else 0.0, args.rs_bits,
                                           args.rs_radius, args.rs_global_p, temp,
                                           bool(args.rs_neutral))
                heat[seg] = 0.8 * heat[seg] + 0.2 * (max(0.0, h0 - win.hval) / seg.numel())
            nun[op] += 1
            r = max(0.0, h0 - win.hval) / max(1e-3, time.time() - tu)
            opq[op] = 0.8 * opq[op] + 0.2 * r                        # credit assignment (EMA)
            prog(op, rnd, u, budget, sum(cnt.values()))
        heat *= 0.97                                                 # priorities age out
        agg = {o: sum(v for k, v in cnt.items() if k.split("@")[0] == o) for o in CHOPS}
        bits, rews = agg["cd-tt"], agg["cd-cn"]
        shares = cnt["cd-sh"]
        rsa = agg["rs-cn"] + agg["rs-tt"] + agg["cd-cf"] + agg["cd-st"] + agg["cd-sg"] \
            + agg["cd-cl"] + agg["cd-rb"] + cnt["rs-sh"] + cnt["sp"]
        va = te = float("nan")
        if rnd % args.val_every == 0:
            va, te = win.evaluate(Xva, vy, args.pass_rows), win.evaluate(Xte, ey, args.pass_rows)
            if args.ckpt:
                torch.save({k: getattr(win, k).cpu() for k in
                            ("base", "conn", "tt", "coef", "deg", "alive", "owner", "step",
                             "sgn", "ocls")}
                           | {"round": rnd, "args": vars(args)}, args.ckpt)
        mc, xc, sf = win.copy_stats()
        print(f"{rnd:>5} | {bits:>9} | {rews:>8} | {shares:>6} | {rsa:>5} | {mc:5.2f} | "
              f"{win.train_acc():6.2f} | {win.hval/win.D:6.3f} | {va:6.2f} | "
              f"{te:6.2f} | {(time.time()-t0)/60:6.1f}"
              + (f"  maxcp={xc} shared={sf:.1f}%" if xc > 1 else ""), flush=True)
        if spath:                                                    # per-round distributions
            al = win.alive
            rec = {"round": rnd, "train": round(win.train_acc(), 2),
                   "val": None if va != va else round(va, 2),
                   "test": None if te != te else round(te, 2),
                   "hinge": round(win.hval / win.D, 4),
                   "ttbits": bits, "rewires": rews, "shares": shares, "rs": rsa,
                   "coefs": agg["cd-cf"], "steps": agg["cd-st"], "signs": agg["cd-sg"],
                   "clsmoves": agg["cd-cl"],
                   "rebuilds": agg["cd-rb"], "explore": round(e, 3),
                   "op_n": nun, "op_q": {k: round(v, 3) for k, v in opq.items()},
                   "gates_layer": torch.bincount(win._lay(al.nonzero().flatten()),
                                                 minlength=win.L).tolist(),
                   "copies_hist": torch.bincount(win.deg[al].long().sum(1),
                                                 minlength=12).tolist(),
                   "ttpop_hist": torch.bincount(win.tt[al].sum(1).long(),
                                                minlength=win.TT + 1).tolist(),
                   "ttbit_ones": win.tt[al].long().sum(0).tolist(),
                   "coef_zero": round(float((win.coef[al] == 0).float().mean()), 4),
                   "gates": int(al.sum()),
                   "deg_mean": [round(v, 4) for v in win.deg[al].float().mean(0).tolist()],
                   "min": round((time.time() - t0) / 60, 1)}
            with open(spath, "a") as f:
                f.write(json.dumps(rec) + "\n")
        if args.check:
            print(f"    state check: max|diff|={win.check():.1f}  "
                  f"hinge drift={abs(win._hinge(win.score) - win.hval):.4f}", flush=True)


if __name__ == "__main__":
    main()
