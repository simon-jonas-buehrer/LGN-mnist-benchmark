"""sbuehrer/hebbian: supervised Hebbian / three-factor learning for a LUT net.

The idea is deliberately not backpropagation.  Each gate has only a 4-entry truth table and fixed
fan-in-2 butterfly wiring.  Training presents an image and its label, then updates only the active
truth-table entry of each gate:

    eligibility  = 1[gate saw local input pattern p]              # pre/post local event
    third factor = target_class_code - predicted_class_mixture    # clamped minus free phase
    Delta T[p]  += eligibility * third factor

That is the supervised version of the "cells that fire together" story: the local input pattern
marks the table entry that is eligible to change, and a label/error-like modulatory signal decides
whether that entry should become more likely to fire.  The subtraction is a small contrastive
Hebbian trick.  If the network already puts all probability on the right class, the clamped target
and free prediction match and the update vanishes.

Hidden layers do not receive class gradients or random feedback matrices.  Instead each class owns
a fixed sparse target assembly in every hidden layer.  A hidden gate therefore learns the mapping

    local two-bit pattern -> should this gate participate in the label's assembly?

The readout layer uses the normal one-vs-rest class groups.  All updates are count/scatter_add
operations under no_grad(); no autograd graph is built and `.backward()` is never called.  The
forward pass is exact bits throughout, so the validation accuracy printed here is the accuracy of
the circuit that `emit_verilog()` writes.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from mnistbench.data import Mnist, N_CLASSES, N_PIXELS
from mnistbench.hw import emit_lutnet, even_thresholds
from mnistbench.spec import Submission


TITLE = "hebbian (target-clamped local plasticity)"

# Same fixed butterfly family as the DFA record, but with a different local update.  The knob is
# `layers`, not `depth`: `bench.run_point` merges the measured netlist fields over the POINTS dict
# and one of them is `depth`, so a POINTS key of that name is silently overwritten in results.json.
#
# The ladder is shallow with a wide readout because that is what the sweep measured (see README):
# this rule gets its accuracy from the width of the grouped vote, not from body depth.  `epochs` is
# a ceiling: validation early-stopping decides where each point actually stops.
POINTS = [
    {"name": "xs", "bits": 1, "width": 512, "layers": 3, "readout": 640, "epochs": 80},
    {"name": "s", "bits": 1, "width": 1024, "layers": 3, "readout": 2560, "epochs": 80},
    {"name": "m", "bits": 1, "width": 2048, "layers": 3, "readout": 5120, "epochs": 80},
    {"name": "l", "bits": 1, "width": 4096, "layers": 3, "readout": 10240, "epochs": 60},
    {"name": "xl", "bits": 1, "width": 8192, "layers": 3, "readout": 20480, "epochs": 60},
]


def _t(a: np.ndarray, device: str) -> torch.Tensor:
    """The harness speaks numpy; torch starts here."""
    return torch.from_numpy(np.ascontiguousarray(a)).to(device)


def _log2(n: int) -> int:
    if n & (n - 1):
        raise ValueError(f"{n} is not a power of two")
    return n.bit_length() - 1


def _butterfly_src(in_dim: int, out_dim: int, stage: int) -> torch.Tensor:
    """(2, out_dim) local source ids into `in_dim`, using the FFT/butterfly tap."""
    j = torch.arange(out_dim)
    if in_dim == out_dim:
        k = stage % _log2(in_dim)
        return torch.stack([j, j ^ (1 << k)])
    if out_dim > in_dim:
        return torch.stack([(2 * j) % in_dim, (2 * j + 1) % in_dim])
    a = (j * in_dim) // out_dim
    return torch.stack([a, (a + in_dim // 2) % in_dim])


# pass-A truth table indexed by p = 2a+b: T[p] = a -> [0,0,1,1] = tt 0b1100.
_RES_TT = torch.tensor([0, 0, 1, 1], dtype=torch.uint8)

# Latent magnitude at init: small enough that the first updates can flip a table entry.
_INIT = 0.02


class HebbianNet:
    """Fixed butterfly wiring; per-gate LUT tables learned by target-clamped local counts."""

    def __init__(
        self,
        bits: int,
        width: int,
        layers: int,
        readout: int,
        device: str,
        g: torch.Generator,
        assembly_frac: float = 0.3,
    ) -> None:
        if readout % N_CLASSES:
            raise ValueError(f"readout {readout} must be divisible by {N_CLASSES}")
        _log2(width)
        self.bits = bits
        self.thresholds = even_thresholds(bits)
        self.n_in = N_PIXELS * bits
        self.device = device
        self.widths = [width] * layers + [readout]
        self.tau = max(1.0, (readout // N_CLASSES) ** 0.5)

        self.offs = [self.n_in]
        in_dim, in_base = self.n_in, 0
        self.srcs: list[torch.Tensor] = []
        for l, w in enumerate(self.widths):
            src = _butterfly_src(in_dim, w, l - 1).contiguous()
            self.srcs.append((src + in_base).to(device))
            in_base = self.offs[-1]
            self.offs.append(self.offs[-1] + w)
            in_dim = w

        # Latents are signed table preferences.  The emitted table is always hard bits.
        base = (_RES_TT.float().to(device) * 2 - 1) * _INIT
        self.Lat = [base.expand(w, 4).contiguous().clone() for w in self.widths]
        self.T = [(lat > 0).to(torch.uint8) for lat in self.Lat]

        self.codes = self._make_codes(readout, assembly_frac, g)

    @property
    def n_sig(self) -> int:
        return self.offs[-1]

    def _make_codes(
        self, readout: int, assembly_frac: float, g: torch.Generator
    ) -> list[torch.Tensor]:
        """Fixed class assemblies, one (10,width) binary codebook per layer."""
        codes: list[torch.Tensor] = []
        k = round(N_CLASSES * assembly_frac)
        k = max(1, min(N_CLASSES - 1, k))
        for w in self.widths[:-1]:
            r = torch.rand((N_CLASSES, w), generator=g, device=self.device)
            idx = r.topk(k, dim=0).indices
            code = torch.zeros((N_CLASSES, w), dtype=torch.float32, device=self.device)
            code.scatter_(0, idx, 1.0)
            codes.append(code)

        readout_code = torch.zeros((N_CLASSES, readout), dtype=torch.float32, device=self.device)
        group = readout // N_CLASSES
        for c in range(N_CLASSES):
            readout_code[c, c * group : (c + 1) * group] = 1.0
        codes.append(readout_code)
        return codes

    def forward(self, enc: torch.Tensor) -> torch.Tensor:
        """enc (n_in, B) uint8 -> acts (n_sig, B) uint8.  Exact boolean forward pass."""
        acts = torch.zeros((self.n_sig, enc.shape[1]), dtype=torch.uint8, device=enc.device)
        acts[: self.n_in] = enc
        for l, s in enumerate(self.srcs):
            p = (acts[s[0]].long() << 1) | acts[s[1]].long()
            acts[self.offs[l] : self.offs[l + 1]] = self.T[l].gather(1, p)
        return acts

    def votes(self, acts: torch.Tensor) -> torch.Tensor:
        out = acts[self.offs[-2] : self.offs[-1]]
        return out.reshape(N_CLASSES, -1, out.shape[1]).sum(1).T.float()

    def tt(self) -> list[torch.Tensor]:
        """Pack each (w,4) table into a (w,) 4-bit int: bit p = T[p]."""
        return [(t[:, 0] | (t[:, 1] << 1) | (t[:, 2] << 2) | (t[:, 3] << 3)).cpu()
                for t in self.T]


def _encode(pix: torch.Tensor, net: HebbianNet) -> torch.Tensor:
    """(N,784) uint8 -> (n_in, N) uint8, pixel-major, matching hw.emit_thermometer."""
    thr = torch.tensor(net.thresholds, device=pix.device, dtype=torch.int16)
    bits = pix.to(torch.int16).unsqueeze(-1) > thr
    return bits.reshape(pix.shape[0], -1).T.contiguous().to(torch.uint8)


class HebbianLut(Submission):
    def __init__(
        self,
        bits: int,
        width: int,
        layers: int,
        readout: int,
        epochs: int,
        batch: int = 4096,
        eta: float = 0.35,
        decay: float = 0.01,
        assembly_frac: float = 0.3,
        patience: int = 12,
    ) -> None:
        self.cfg = dict(
            bits=bits,
            width=width,
            layers=layers,
            readout=readout,
            epochs=epochs,
            batch=batch,
            eta=eta,
            decay=decay,
            assembly_frac=assembly_frac,
            patience=patience,
        )
        self.net: HebbianNet | None = None

    @torch.no_grad()
    def train(self, data: Mnist, *, device: str = "cpu", seed: int = 0) -> None:
        c = self.cfg
        torch.manual_seed(seed)
        g = torch.Generator(device=device).manual_seed(seed)
        net = HebbianNet(
            c["bits"], c["width"], c["layers"], c["readout"], device, g, c["assembly_frac"]
        )

        enc_tr = _encode(_t(data.train_x, device), net)
        y_tr = _t(data.train_y, device).long()
        enc_va = _encode(_t(data.val_x, device), net)
        y_va = _t(data.val_y, device).long()

        n = enc_tr.shape[1]
        steps = max(1, n // c["batch"])
        best_val, best_T, best_ep = -1.0, [t.clone() for t in net.T], 0
        t0 = time.time()
        for ep in range(c["epochs"]):
            perm = torch.randperm(n, generator=g, device=device)
            loss = 0.0
            for i in range(steps):
                idx = perm[i * c["batch"] : min((i + 1) * c["batch"], n)]
                loss += self._step(net, enc_tr[:, idx], y_tr[idx])
            loss /= steps

            acc = self._accuracy(net, enc_va, y_va)
            if acc > best_val:
                best_val, best_ep = acc, ep
                best_T = [t.clone() for t in net.T]
            print(
                f"  epoch {ep + 1:3d}/{c['epochs']}  loss {loss:.3f}  val {acc:.2f}%  "
                f"(best {best_val:.2f}% @ {best_ep + 1})  "
                f"{(ep + 1) / (time.time() - t0):.2f} ep/s",
                flush=True,
            )
            if ep - best_ep >= c["patience"]:
                print(f"  early stop at epoch {ep + 1}: no gain since {best_ep + 1}", flush=True)
                break

        net.T = [t.clone() for t in best_T]
        self.net = net

    def _step(self, net: HebbianNet, enc: torch.Tensor, y: torch.Tensor) -> float:
        """One contrastive Hebbian count update.  No autograd and no cross-layer chain rule."""
        c = self.cfg
        B = enc.shape[1]
        acts = net.forward(enc)

        logits = net.votes(acts) / net.tau
        prob = torch.softmax(logits, 1)
        loss = -torch.log(prob[torch.arange(B, device=enc.device), y] + 1e-12).mean()

        for l, w in enumerate(net.widths):
            src = net.srcs[l]
            p = (acts[src[0]].long() << 1) | acts[src[1]].long()
            code = net.codes[l]
            clamped = code[y].T                         # (w,B): target-label assembly
            free = (prob @ code).T                      # (w,B): prediction-weighted assembly
            third = clamped - free

            idx = torch.arange(w, device=enc.device)[:, None] * 4 + p
            num = torch.zeros(w * 4, device=enc.device)
            den = torch.zeros(w * 4, device=enc.device)
            num.scatter_add_(0, idx.reshape(-1), third.reshape(-1))
            den.scatter_add_(0, idx.reshape(-1), torch.ones(w * B, device=enc.device))
            update = num.view(w, 4) / (den.view(w, 4) + 1.0)

            net.Lat[l].mul_(1.0 - c["decay"]).add_(update, alpha=c["eta"])
            net.T[l] = (net.Lat[l] > 0).to(torch.uint8)

        return loss.item()

    def _chunk(self) -> int:
        c = self.cfg
        n_sig = N_PIXELS * c["bits"] + c["width"] * c["layers"] + c["readout"]
        return max(64, min(4096, 2**28 // n_sig))

    @torch.no_grad()
    def _accuracy(self, net: HebbianNet, enc: torch.Tensor, y: torch.Tensor) -> float:
        ch, right = self._chunk(), 0
        for i in range(0, enc.shape[1], ch):
            v = net.votes(net.forward(enc[:, i : i + ch]))
            right += (v.argmax(1) == y[i : i + ch]).sum().item()
        return right / enc.shape[1] * 100

    @torch.no_grad()
    def predict(self, pix: np.ndarray) -> np.ndarray:
        net = self.net
        enc = _encode(_t(pix, net.device), net)
        ch = self._chunk()
        out = [net.votes(net.forward(enc[:, i : i + ch])).argmax(1).cpu()
               for i in range(0, enc.shape[1], ch)]
        return torch.cat(out).numpy()

    @torch.no_grad()
    def scores(self, pix: np.ndarray) -> np.ndarray:
        net = self.net
        group = net.widths[-1] // N_CLASSES
        enc = _encode(_t(pix, net.device), net)
        ch = self._chunk()
        out = [(net.votes(net.forward(enc[:, i : i + ch])) / group).cpu()
               for i in range(0, enc.shape[1], ch)]
        return torch.cat(out).numpy()

    def emit_verilog(self) -> str:
        net = self.net
        layers = [(s[0].cpu(), s[1].cpu(), tt) for s, tt in zip(net.srcs, net.tt())]
        return emit_lutnet(net.thresholds, layers)


def build(**point) -> Submission:
    return HebbianLut(**point)
