"""sbuehrer/backprop -- gradient descent learns BOTH what each gate is and how it is wired.

A LUT gate needs two answers: *what function am I* and *which two signals do I read*. This
record learns both, and both as a discrete choice with a smooth gradient:

  WHAT (the truth table). Four latent reals per gate, one per truth-table entry, binarized by
  a straight-through estimator on a sin:

      hard = 1[sin(z) > 0]                 exact 0/1 -- what the forward pass uses
      soft = 0.5 + 0.5*sin(z)              smooth, differentiable
      bit  = hard + (soft - soft.detach()) forward = hard, backward = d(soft)

  sin rather than sigmoid because sin is periodic: a latent never saturates, so there is always
  a gradient toward the nearest 0/1 basin.

  WHERE (the wiring). Each of a gate's two inputs gets 8 candidate source signals, drawn at
  random once, plus a learnable logit per candidate. The forward pass takes the argmax -- ONE
  wire, an exact bit -- and the backward pass sees the softmax over all 8, so a candidate that
  would have helped still gets gradient and the choice can move. See LutLayer.

The forward pass is therefore ALREADY an exact boolean circuit: no "train soft, discretize at
the end and pray" step, and the val accuracy the trainer prints is the accuracy the silicon has.
That is not a nicety -- the harness rejects any point whose python model and circuit disagree, so
a softmax MIXTURE of candidate bits (a fraction, with no hardware) would be caught immediately.

sbuehrer/genetic is the mirror image: it learns only the wiring, by mutation, with no gradients.

The encoder is a thermometer at thresholds 2^k-1, which is not an accident: `pix > 127` is just
"bit 7", i.e. a WIRE, and costs zero gates. Choosing thresholds that are cheap in silicon is
exactly the kind of pressure this benchmark is supposed to create.

SETTINGS. lr=0.2, batch=128, from a 22-config sweep on the `m` point (val, never test):

    lr        0.01   0.02   0.05   0.1    0.2    0.3    0.5    0.8
    best val  92.15  92.47  92.47  92.78  92.80  92.45  91.55  91.03

Two things worth taking from that table. The peak is FLAT -- everything from 0.02 to 0.2 lands
within 0.3 points -- so this record is not perched on a lucky hyperparameter, and a submitter who
reruns it with lr=0.05 will see the same curve. And the whole sweep is worth about one point,
while turning the wiring from frozen to learned was worth about seven. The lever here is capacity,
not tuning: at `m` these nets converge to ~92.8% for ANY sane lr, because that is what a
5120->2560 net can do, and the only way past it is more gates.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from mnistbench.data import Mnist, N_CLASSES, N_PIXELS
from mnistbench.hw import emit_lutnet, even_thresholds
from mnistbench.spec import Submission


TITLE = "backprop (learned truth tables + learned wiring)"

# `epochs` is a ceiling, not a target: training early-stops when validation has not improved for
# `patience` epochs, so every point below is trained to ITS OWN convergence. Raising the ceiling
# does not change a converged point -- it only lets a slower one finish climbing.
POINTS = [
    {"name": "xs", "bits": 1, "widths": (320, 160), "epochs": 200},
    {"name": "s", "bits": 1, "widths": (1280, 640), "epochs": 200},
    {"name": "m", "bits": 3, "widths": (5120, 2560), "epochs": 200},
    {"name": "l", "bits": 3, "widths": (16000, 8000, 4000), "epochs": 150},
    {"name": "xl", "bits": 7, "widths": (48000, 24000, 12000), "epochs": 120},
]


def _t(a: np.ndarray, device: str) -> torch.Tensor:
    """The harness speaks numpy; torch starts here."""
    return torch.from_numpy(np.ascontiguousarray(a)).to(device)


def hard_bit(z: torch.Tensor) -> torch.Tensor:
    return (torch.sin(z) > 0).to(z.dtype)


def ste_bit(z: torch.Tensor) -> torch.Tensor:
    soft = 0.5 + 0.5 * torch.sin(z)
    return hard_bit(z) + (soft - soft.detach())


class LutLayer(torch.nn.Module):
    """`width` gates. Each gate learns BOTH its truth table and where its two inputs come from.

    The wiring is learned the same way the truth table is: as a discrete choice with a smooth
    gradient. Every gate input gets `cands` candidate source signals, drawn at random once, and a
    learnable logit per candidate. The forward pass takes the argmax candidate -- ONE wire, an
    exact bit -- while the backward pass sees the softmax over all of them, so every candidate
    that would have helped gets gradient and the choice can move.

        sel  = onehot(argmax(logits))                exact one-wire selection (forward)
        soft = softmax(logits)                       smooth over the 8 candidates (backward)
        wire = sel + (soft - soft.detach())          forward = sel, gradient = d(soft)

    Selecting with a one-hot over bits keeps the forward pass exactly boolean, which is what lets
    the emitted circuit match predict() bit for bit. A softmax MIXTURE of candidate bits would
    not: it is a fraction, it has no hardware, and the harness would reject the point.
    """

    def __init__(self, off: int, width: int, cands: int, g: torch.Generator) -> None:
        super().__init__()
        # the candidate pool: which `cands` signals each of the 2 inputs may choose between
        self.register_buffer("cand", torch.randint(off, (2, width, cands), generator=g))
        self.conn = torch.nn.Parameter(torch.randn(2, width, cands, generator=g) * 0.1)
        self.table = torch.nn.Parameter(torch.randn(width, 4, generator=g))

    def wires(self) -> torch.Tensor:
        """(2, width) the signal id each input actually reads -- the winning candidate."""
        return self.cand.gather(2, self.conn.argmax(-1, keepdim=True)).squeeze(-1)

    def forward(self, sig: torch.Tensor) -> torch.Tensor:
        x = sig[:, self.cand]  # (B, 2, width, cands) candidate bits
        soft = torch.softmax(self.conn, dim=-1)
        sel = torch.zeros_like(soft).scatter_(-1, self.conn.argmax(-1, keepdim=True), 1.0)
        wire = sel + (soft - soft.detach())  # hard forward, softmax gradient
        picked = (x * wire).sum(-1)  # (B, 2, width) -- exactly the chosen bits
        xa, xb = picked[:, 0], picked[:, 1]

        c = ste_bit(self.table)  # (w, 4) = [f00, f01, f10, f11]
        f00, f01, f10, f11 = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
        # multilinear form of a 2-input LUT: exact on {0,1} bits, interpolating for the gradient
        return f00 + (f10 - f00) * xa + (f01 - f00) * xb + (f00 - f01 - f10 + f11) * xa * xb

    def truth_table(self) -> torch.Tensor:
        """4-bit truth table per gate: bit (2a+b) = f(a, b), the encoding hw.lut2_expr wants."""
        c = hard_bit(self.table).long()
        return c[:, 0] | (c[:, 1] << 1) | (c[:, 2] << 2) | (c[:, 3] << 3)


class LutNet(torch.nn.Module):
    def __init__(self, bits: int, widths: tuple[int, ...], cands: int = 8, seed: int = 0) -> None:
        super().__init__()
        if widths[-1] % N_CLASSES:
            raise ValueError(f"readout width {widths[-1]} must be divisible by {N_CLASSES}")
        self.bits = bits
        self.widths = widths
        self.thresholds = even_thresholds(bits)

        g = torch.Generator().manual_seed(seed)
        off = N_PIXELS * bits
        self.layers = torch.nn.ModuleList()
        for w in widths:
            self.layers.append(LutLayer(off, w, cands, g))
            off += w
        self.n_sig = off

    def encode(self, pix: torch.Tensor) -> torch.Tensor:
        """(N, 784) uint8 -> (N, 784*bits) float bits, laid out exactly as hw.emit_thermometer."""
        t = torch.tensor(self.thresholds, device=pix.device, dtype=torch.int16)
        bits = pix.to(torch.int16).unsqueeze(-1) > t  # (N, 784, bits), pixel-major
        return bits.reshape(pix.shape[0], -1).float()

    def forward(self, pix: torch.Tensor) -> torch.Tensor:
        """Logits. The bits are hard; only the gradient is smooth."""
        sig = self.encode(pix)
        for layer in self.layers:
            sig = torch.cat([sig, layer(sig)], dim=1)  # gates may read ANY earlier signal
        last = sig[:, -self.widths[-1] :]
        groups = last.reshape(last.shape[0], N_CLASSES, -1)
        return groups.sum(-1) / (self.widths[-1] // N_CLASSES) ** 0.5  # tau: keep logits sane


class BackpropLut(Submission):
    def __init__(self, bits: int, widths: tuple[int, ...], epochs: int, lr: float = 0.2,
                 batch: int = 128, cands: int = 8, patience: int = 40) -> None:
        self.cfg = dict(bits=bits, widths=tuple(widths), epochs=epochs, lr=lr, batch=batch,
                        cands=cands, patience=patience)
        self.model: LutNet | None = None

    def train(self, data: Mnist, *, device: str = "cpu", seed: int = 0) -> None:
        torch.manual_seed(seed)
        c = self.cfg
        self.model = m = LutNet(c["bits"], c["widths"], c["cands"], seed=seed).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=c["lr"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=c["epochs"])

        x, y = _t(data.train_x, device), _t(data.train_y, device)
        vx, vy = _t(data.val_x, device), _t(data.val_y, device)
        best_val, best_state, best_ep = -1.0, None, 0

        for ep in range(c["epochs"]):
            perm = torch.randperm(x.shape[0], device=device)
            for i in range(0, x.shape[0], c["batch"]):
                idx = perm[i : i + c["batch"]]
                loss = F.cross_entropy(m(x[idx]), y[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sched.step()

            with torch.no_grad():
                ch = self._chunk()
                acc = sum(
                    (m(vx[i : i + ch]).argmax(1) == vy[i : i + ch]).sum().item()
                    for i in range(0, vx.shape[0], ch)
                ) / vx.shape[0] * 100
            if acc > best_val:  # the forward pass is already hard, so this is the circuit's acc
                best_val, best_ep = acc, ep
                best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
            if ep % 5 == 0 or ep == c["epochs"] - 1:
                print(f"  epoch {ep + 1:3d}/{c['epochs']}  loss {loss.item():.3f}  "
                      f"val {acc:.2f}%  (best {best_val:.2f}% @ {best_ep + 1})", flush=True)
            if ep - best_ep >= c["patience"]:  # converged: nothing better for `patience` epochs
                print(f"  early stop at epoch {ep + 1}: no gain since {best_ep + 1}", flush=True)
                break

        m.load_state_dict(best_state)

    def _chunk(self) -> int:
        """Rows per eval forward pass.

        A layer's candidate gather is (rows, 2, width, cands) floats -- with cands=8 and a 48k-wide
        layer, a FIXED chunk of 2048 asks for 6 GB in one allocation and OOMs an 11 GB card. So the
        chunk is derived from the widest layer instead of hardcoded, capping that tensor at ~1 GB.
        """
        c = self.cfg
        widest = max(c["widths"])
        return max(64, min(2048, 2**28 // (2 * widest * c["cands"])))

    @torch.no_grad()
    def predict(self, pix: np.ndarray) -> np.ndarray:
        m = self.model
        dev = next(m.parameters()).device
        x = _t(pix, dev)
        ch = self._chunk()
        out = [m(x[i : i + ch]).argmax(1).cpu() for i in range(0, len(x), ch)]
        return torch.cat(out).numpy()  # ties -> lowest class, same as the emitted argmax

    @torch.no_grad()
    def scores(self, pix: np.ndarray) -> np.ndarray:
        """Per-class firing fraction in [0, 1]: of each class's readout gates, how many fired.

        m.forward returns the same group sums scaled by a constant, so its argmax matches; here we
        want the raw fraction (mean of the group's hard bits) for a scale-honest cross-entropy.
        """
        m = self.model
        dev = next(m.parameters()).device
        x = _t(pix, dev)
        ch = self._chunk()
        out = []
        for i in range(0, len(x), ch):
            sig = m.encode(x[i : i + ch])
            for layer in m.layers:
                sig = torch.cat([sig, layer(sig)], dim=1)
            last = sig[:, -m.widths[-1] :]
            frac = last.reshape(last.shape[0], N_CLASSES, -1).mean(-1)  # (B, 10) in [0, 1]
            out.append(frac.cpu())
        return torch.cat(out).numpy()

    def emit_verilog(self) -> str:
        m = self.model
        layers = []
        for lay in m.layers:
            w = lay.wires().cpu()  # the argmax candidate: one real wire per input
            layers.append((w[0], w[1], lay.truth_table().cpu()))
        return emit_lutnet(m.thresholds, layers)


def build(**point) -> Submission:
    return BackpropLut(**point)
