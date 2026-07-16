"""sbuehrer/targetprop: fixed Monarch wiring, gate truth tables learned WITHOUT gradients.

The other two records learn the wiring. This one fixes it in a Monarch pattern (block-diagonal
within groups on even layers, across groups on odd layers) so the receptive field reaches every
output in log-depth, and learns only the 2-input truth table of each gate.

No gradients. Each iteration:

  1. run the discrete net on a large batch, caching every gate's input pattern p = 2a+b;
  2. at the readout, ask each class group for a few bit flips that would widen the margin between
     the true class and the best wrong class (margin, not accuracy, so the signal is a slope);
  3. propagate those targets down. A gate with target t and current pattern p either already
     outputs t (nothing to do), or it does not, in which case we both (a) VOTE that its table
     entry T[p] should be t, and (b) push a target upstream to the input that, if flipped, would
     make the current table output t (minimal-flip; prefer the mixing input so the pass-through
     path is preserved). Change the gate or change its inputs, decided by the votes.
  4. after the whole batch, update each table entry by majority vote, through a small real-valued
     accumulator so a bit only flips once a consistent vote builds up. The emitted table stays
     exactly boolean, so the circuit matches predict() bit for bit.

Residual init (every gate passes input A) means the net starts as identity; the accumulator keeps
the latents unsaturated so the vote can move them (a plain argmax update would sit stuck at
identity forever).
"""

from __future__ import annotations

import time

import numpy as np
import torch

from mnistbench.data import Mnist, N_CLASSES, N_PIXELS
from mnistbench.hw import emit_lutnet, even_thresholds
from mnistbench.spec import Submission

TITLE = "targetprop (fixed Monarch wiring, gradient-free counting)"

TOPO_SEED = 0

POINTS = [
    {"name": "xs", "bits": 1, "width": 1024, "depth": 20, "readout": 320},
    {"name": "s", "bits": 1, "width": 2048, "depth": 22, "readout": 640},
    {"name": "m", "bits": 3, "width": 4096, "depth": 24, "readout": 640},
    {"name": "l", "bits": 3, "width": 8192, "depth": 26, "readout": 1280},
]


def _t(a: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).to(device)


def _monarch_groups(in_dim: int, out_dim: int) -> int:
    g = 256
    while g > 1 and (in_dim % g or out_dim % g or in_dim // g < 2):
        g //= 2
    if g < 2:
        raise ValueError(f"no monarch grouping for {in_dim}->{out_dim}")
    return g


def _monarch_src(in_dim: int, out_dim: int, parity: int) -> torch.Tensor:
    """(out_dim, 2) local source indices into `in_dim`, fan-in 2, deterministic Monarch tap."""
    g = _monarch_groups(in_dim, out_dim)
    ipg, opg = in_dim // g, out_dim // g
    j = torch.arange(out_dim)
    r, c = j // opg, j % opg
    t = torch.arange(2)
    if parity == 0:  # within-group: block-diagonal factor
        mon = r[:, None] * ipg + (c[:, None] * 2 + t[None]) % ipg
    else:            # across-group: the transpose factor
        mon = ((r[:, None] + t[None] * (g // 2)) % g) * ipg + (c % ipg)[:, None]
    return mon  # (out_dim, 2)


# pass-A truth table indexed by p = 2a+b:  T[p] = (p>>1)&1 = a  ->  [0,0,1,1] = tt 0b1100
_RES_TT = torch.tensor([0, 0, 1, 1], dtype=torch.uint8)


class MonarchNet:
    """Fixed Monarch fan-in-2 wiring; per-gate 4-entry truth table T (learned), soft accumulator Lat."""

    def __init__(self, bits: int, width: int, depth: int, readout: int, device: str) -> None:
        if readout % N_CLASSES:
            raise ValueError(f"readout {readout} must be divisible by {N_CLASSES}")
        self.bits = bits
        self.thresholds = even_thresholds(bits)
        self.n_in = N_PIXELS * bits
        self.device = device
        self.widths = [width] * depth + [readout]

        # fixed wiring: srcs[l] = (2, w_l) GLOBAL ids; layer l reads only layer l-1 (or encoder)
        self.offs = [self.n_in]
        self.in_base = [0]  # global id where layer l's inputs start (encoder for l=0)
        in_dim = self.n_in
        in_base = 0
        self.srcs: list[torch.Tensor] = []
        for l, w in enumerate(self.widths):
            mon = _monarch_src(in_dim, w, l % 2).T.contiguous()  # (2, w) local into in_dim
            self.srcs.append((mon + in_base).to(device))
            in_base = self.offs[-1]                # next layer reads THIS layer's outputs
            self.in_base.append(in_base)
            self.offs.append(self.offs[-1] + w)
            in_dim = w

        # residual init: every gate passes input A
        self.T = [_RES_TT.clone().to(device).expand(w, 4).contiguous() for w in self.widths]
        self.Lat = [((_RES_TT.float().to(device) * 2 - 1) * 0.05).expand(w, 4).contiguous()
                    for w in self.widths]

    @property
    def n_sig(self) -> int:
        return self.offs[-1]

    def forward(self, enc: torch.Tensor) -> torch.Tensor:
        """enc (n_in, B) uint8 -> acts (n_sig, B) uint8. Layered: layer l reads earlier ids only."""
        B = enc.shape[1]
        acts = torch.zeros((self.n_sig, B), dtype=torch.uint8, device=enc.device)
        acts[: self.n_in] = enc
        for l, s in enumerate(self.srcs):
            a = acts[s[0]]                       # (w, B)
            b = acts[s[1]]
            p = (a.long() << 1) | b.long()       # (w, B) in {0,1,2,3}
            out = self.T[l].gather(1, p)         # (w, B) uint8
            acts[self.offs[l] : self.offs[l + 1]] = out
        return acts

    def votes(self, acts: torch.Tensor) -> torch.Tensor:
        out = acts[self.offs[-2] : self.offs[-1]]  # (R, B)
        return out.reshape(N_CLASSES, -1, out.shape[1]).sum(1).T.float()  # (B, 10)

    def tt(self) -> list[torch.Tensor]:
        """pack each (w,4) table into a (w,) 4-bit int: bit p = T[p]."""
        return [(t[:, 0] | (t[:, 1] << 1) | (t[:, 2] << 2) | (t[:, 3] << 3)).cpu()
                for t in self.T]


def _encode(pix: torch.Tensor, net: MonarchNet) -> torch.Tensor:
    """(N,784) uint8 -> (n_in, N) uint8, pixel-major, matching hw.emit_thermometer."""
    thr = torch.tensor(net.thresholds, device=pix.device, dtype=torch.int16)
    bits = pix.to(torch.int16).unsqueeze(-1) > thr  # (N, 784, bits)
    return bits.reshape(pix.shape[0], -1).T.contiguous().to(torch.uint8)


class TargetPropLut(Submission):
    def __init__(self, bits: int, width: int, depth: int, readout: int, iters: int = 4000,
                 batch: int = 8192, margin: int = 2, eta: float = 0.5, patience: int = 20,
                 eval_every: int = 50) -> None:
        self.cfg = dict(bits=bits, width=width, depth=depth, readout=readout, iters=iters,
                        batch=batch, margin=margin, eta=eta, patience=patience,
                        eval_every=eval_every)
        self.net: MonarchNet | None = None

    def train(self, data: Mnist, *, device: str = "cpu", seed: int = 0) -> None:
        c = self.cfg
        torch.manual_seed(seed)
        net = MonarchNet(c["bits"], c["width"], c["depth"], c["readout"], device)

        enc_tr = _encode(_t(data.train_x, device), net)  # (n_in, N)
        y_tr = _t(data.train_y, device).long()
        enc_va = _encode(_t(data.val_x, device), net)
        y_va = _t(data.val_y, device).long()
        g = torch.Generator(device=device).manual_seed(seed)

        best_val, best_T, stale = -1.0, [t.clone() for t in net.T], 0
        t0 = time.time()
        for it in range(c["iters"]):
            idx = torch.randint(enc_tr.shape[1], (c["batch"],), generator=g, device=device)
            self._step(net, enc_tr[:, idx], y_tr[idx], c["eta"])
            if (it + 1) % c["eval_every"] == 0 or it + 1 == c["iters"]:
                acc = self._accuracy(net, enc_va, y_va)
                if acc > best_val:
                    best_val, best_T, stale = acc, [t.clone() for t in net.T], 0
                else:
                    stale += 1
                print(f"  iter {it + 1:5d}/{c['iters']}  val {acc:.2f}%  "
                      f"(best {best_val:.2f}%, stale {stale})  "
                      f"{(it + 1) / (time.time() - t0):.1f} it/s", flush=True)
                if stale >= c["patience"]:
                    print(f"  early stop at iter {it + 1}: converged (best {best_val:.2f}%)",
                          flush=True)
                    break
        for t, bt in zip(net.T, best_T):
            t.copy_(bt)
        self.net = net

    # ---- the counting update -------------------------------------------------------------
    def _step(self, net: MonarchNet, enc: torch.Tensor, y: torch.Tensor, eta: float) -> None:
        B = enc.shape[1]
        acts = net.forward(enc)

        # readout targets: widen the margin of the true class over the best wrong class
        v = net.votes(acts)  # (B, 10)
        R = net.widths[-1]
        g = R // N_CLASSES
        wrong = v.clone().scatter_(1, y[:, None], -1e9)
        r = wrong.argmax(1)                                   # best wrong class
        deficit = (v.gather(1, r[:, None]) - v.gather(1, y[:, None])).squeeze(1) + self.cfg["margin"]
        active = deficit > 0                                  # samples on the wrong side
        q = torch.clamp(torch.ceil(deficit / 2), min=1).long()  # bits to flip each side

        # target buffer for the current layer's outputs: -1 don't-care, else desired bit
        tgt = torch.full((R, B), -1, dtype=torch.int8, device=enc.device)
        out_last = acts[net.offs[-2] : net.offs[-1]]         # (R, B)
        self._request_flips(out_last, tgt, y, q, active, g, want=1)   # true class: 0 -> 1
        self._request_flips(out_last, tgt, r, q, active, g, want=0)   # wrong class: 1 -> 0

        # propagate down, accumulating table votes; update at the end
        V = [torch.zeros(w, 4, 2, device=enc.device) for w in net.widths]
        for l in range(len(net.widths) - 1, -1, -1):
            s = net.srcs[l]
            a = acts[s[0]]                                    # (w, B) uint8
            b = acts[s[1]]
            p = ((a.long() << 1) | b.long())                 # (w, B) in {0,1,2,3}
            care = tgt >= 0
            t = tgt.clamp(min=0).long()                       # desired output where care
            w_idx = torch.arange(net.widths[l], device=enc.device)[:, None].expand_as(p)
            flat = (w_idx * 8 + p * 2 + t)[care]
            V[l].view(-1).scatter_add_(0, flat, torch.ones(flat.shape[0], device=enc.device))
            if l == 0:
                break
            tgt = self._propagate(net, l, p, t, care, a, b, s)  # -> targets for layer l-1

        # apply the majority vote through the soft accumulator
        for l in range(len(net.widths)):
            diff = (V[l][:, :, 1] - V[l][:, :, 0]) / (V[l].sum(-1) + 1.0)  # (w,4) in [-1,1]
            net.Lat[l] += eta * diff
            net.T[l] = (net.Lat[l] > 0).to(torch.uint8)

    def _request_flips(self, out, tgt, cls, q, active, g, want):
        """For each active sample, mark `q` bits of class group `cls` that currently != want."""
        B = out.shape[1]
        cand = (out != want)                                  # (R, B) bits we could flip
        # restrict to the class's group: rows [cls*g, cls*g+g)
        rows = (torch.arange(out.shape[0], device=out.device)[:, None] // g)  # (R,1) group id
        in_group = rows == cls[None, :]                       # (R, B)
        pick = cand & in_group & active[None, :]
        # cumulative count within group, keep first q per sample
        order = pick.long().cumsum(0)
        keep = pick & (order <= q[None, :])
        tgt[keep] = want

    def _propagate(self, net, l, p, t, care, a, b, s):
        """Return targets (w_prev, B) for layer l-1, from the mismatched gates of layer l."""
        B = p.shape[1]
        w_prev = net.widths[l - 1]
        base = net.in_base[l]                                  # global id where layer l-1 starts
        Tl = net.T[l]                                          # (w,4)
        cur = Tl.gather(1, p)                                  # (w,B) current output
        mismatch = care & (cur != t.to(torch.uint8))
        pprime = _nearest_pattern(Tl, p, t)                   # (w,B) desired pattern or -1
        ok = mismatch & (pprime >= 0)
        a_des = (pprime >> 1) & 1
        b_des = pprime & 1
        # vote desired values onto the previous layer's signals (local ids)
        acc = torch.zeros((w_prev, B), dtype=torch.float32, device=p.device)   # signed vote
        cnt = torch.zeros((w_prev, B), dtype=torch.float32, device=p.device)
        for src, des, cur_v in ((s[0], a_des, a.long()), (s[1], b_des, b.long())):
            push = ok & (des != cur_v)                         # only the input that must change
            sign = des.to(torch.float32) * 2 - 1               # +1 want 1, -1 want 0
            loc = (src - base)[:, None].expand(-1, B)          # (w,B) local id in [0,w_prev)
            acc.scatter_add_(0, loc.long(), sign * push.float())
            cnt.scatter_add_(0, loc.long(), push.float())
        maj = (acc > 0).to(torch.int8)
        return torch.where(cnt > 0, maj, torch.full_like(maj, -1))

    # ---- eval ----------------------------------------------------------------------------
    @torch.no_grad()
    def _accuracy(self, net, enc, y, chunk: int = 4096):
        right = 0
        for i in range(0, enc.shape[1], chunk):
            v = net.votes(net.forward(enc[:, i : i + chunk]))
            right += (v.argmax(1) == y[i : i + chunk]).sum().item()
        return right / enc.shape[1] * 100

    @torch.no_grad()
    def predict(self, pix: np.ndarray) -> np.ndarray:
        net = self.net
        enc = _encode(_t(pix, net.device), net)
        out = []
        for i in range(0, enc.shape[1], 4096):
            out.append(net.votes(net.forward(enc[:, i : i + 4096])).argmax(1).cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def scores(self, pix: np.ndarray) -> np.ndarray:
        net = self.net
        g = net.widths[-1] // N_CLASSES
        enc = _encode(_t(pix, net.device), net)
        out = []
        for i in range(0, enc.shape[1], 4096):
            out.append((net.votes(net.forward(enc[:, i : i + 4096])) / g).cpu())
        return torch.cat(out).numpy()

    def emit_verilog(self) -> str:
        net = self.net
        layers = [(s[0].cpu(), s[1].cpu(), tt) for s, tt in zip(net.srcs, net.tt())]
        return emit_lutnet(net.thresholds, layers)


def _nearest_pattern(Tl: torch.Tensor, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """(w,B): for each gate, the pattern p' with T[p']==t nearest to p (Hamming), prefer flip-b."""
    w, B = p.shape
    # search order over the 4 patterns by Hamming distance from p, tie prefer flipping low bit b
    # distance-0: p itself; distance-1: p^1 (flip b), p^2 (flip a); distance-2: p^3
    order = torch.stack([p, p ^ 1, p ^ 2, p ^ 3], 0)         # (4, w, B) candidate patterns
    Tl_exp = Tl[None].expand(4, w, 4)                         # (4, w, 4)
    lut = Tl_exp.gather(2, order)                             # (4, w, B) = T[gate, candidate]
    hits = lut == t.to(torch.uint8)[None]
    first = hits.float().argmax(0)                            # (w,B) first order index that hits
    any_hit = hits.any(0)
    chosen = order.gather(0, first[None]).squeeze(0)
    return torch.where(any_hit, chosen, torch.full_like(chosen, -1))


def build(**point) -> Submission:
    return TargetPropLut(**point)
