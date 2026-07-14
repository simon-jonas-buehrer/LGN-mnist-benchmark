"""sbuehrer/backprop -- learn the truth tables by gradient descent (a difflogic-style LUT net).

The wiring is random and FROZEN; what is learned is what each gate *is*. Every gate has four
latent reals, one per truth-table entry, and a straight-through estimator gives them gradients:

    hard = 1[sin(z) > 0]                 exact 0/1 -- this is what the forward pass uses
    soft = 0.5 + 0.5*sin(z)              smooth, differentiable
    bit  = hard + (soft - soft.detach()) forward = hard, backward = d(soft)

The forward pass is therefore ALREADY an exact boolean circuit -- there is no
"discretize at the end and pray" step, and the accuracy the trainer prints is the accuracy the
silicon has. sin rather than sigmoid because sin is periodic: a latent never saturates, so
there is always a gradient toward the nearest 0/1 basin.

Between the two gates a LUT layer needs (which two signals do I read? what function am I?),
this optimizer answers only the second. sbuehrer/genetic answers only the first. That is the
comparison.

The encoder is a thermometer at thresholds 2^k-1, which is not an accident: `pix > 127` is
just "bit 7", i.e. a WIRE, and costs zero gates. Choosing thresholds that are cheap in silicon
is exactly the kind of pressure this benchmark is supposed to create.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from mnistbench.data import Mnist, N_CLASSES, N_PIXELS
from mnistbench.hw import emit_lutnet, even_thresholds
from mnistbench.spec import Submission


TITLE = "backprop (learned truth tables, frozen random wiring)"

POINTS = [
    {"name": "xs", "bits": 1, "widths": (320, 160), "epochs": 30},
    {"name": "s", "bits": 1, "widths": (1280, 640), "epochs": 30},
    {"name": "m", "bits": 3, "widths": (5120, 2560), "epochs": 40},
    {"name": "l", "bits": 3, "widths": (16000, 8000, 4000), "epochs": 40},
    {"name": "xl", "bits": 7, "widths": (48000, 24000, 12000), "epochs": 50},
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
    """`width` gates, each reading two of the `off` signals that already exist."""

    def __init__(self, off: int, width: int, g: torch.Generator) -> None:
        super().__init__()
        a = torch.randint(off, (width,), generator=g)
        b = torch.randint(off, (width,), generator=g)
        b = torch.where(a == b, (b + 1) % off, b)  # a gate reading x twice is a wasted gate
        self.register_buffer("idx_a", a)  # wiring: fixed at init, never learned
        self.register_buffer("idx_b", b)
        self.table = torch.nn.Parameter(torch.randn(width, 4, generator=g))  # this is what learns

    def forward(self, sig: torch.Tensor) -> torch.Tensor:
        xa, xb = sig[:, self.idx_a], sig[:, self.idx_b]
        c = ste_bit(self.table)  # (w, 4) = [f00, f01, f10, f11]
        f00, f01, f10, f11 = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
        # multilinear form of a 2-input LUT: exact on {0,1} bits, interpolating for the gradient
        return f00 + (f10 - f00) * xa + (f01 - f00) * xb + (f00 - f01 - f10 + f11) * xa * xb

    def truth_table(self) -> torch.Tensor:
        """4-bit truth table per gate: bit (2a+b) = f(a, b), the encoding hw.lut2_expr wants."""
        c = hard_bit(self.table).long()
        return c[:, 0] | (c[:, 1] << 1) | (c[:, 2] << 2) | (c[:, 3] << 3)


class LutNet(torch.nn.Module):
    def __init__(self, bits: int, widths: tuple[int, ...], seed: int = 0) -> None:
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
            self.layers.append(LutLayer(off, w, g))
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
    def __init__(self, bits: int, widths: tuple[int, ...], epochs: int, lr: float = 0.01,
                 batch: int = 256) -> None:
        self.cfg = dict(bits=bits, widths=tuple(widths), epochs=epochs, lr=lr, batch=batch)
        self.model: LutNet | None = None

    def train(self, data: Mnist, *, device: str = "cpu", seed: int = 0) -> None:
        torch.manual_seed(seed)
        c = self.cfg
        self.model = m = LutNet(c["bits"], c["widths"], seed=seed).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=c["lr"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=c["epochs"])

        x, y = _t(data.train_x, device), _t(data.train_y, device)
        vx, vy = _t(data.val_x, device), _t(data.val_y, device)
        best_val, best_state = -1.0, None

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
                acc = sum(
                    (m(vx[i : i + 2048]).argmax(1) == vy[i : i + 2048]).sum().item()
                    for i in range(0, vx.shape[0], 2048)
                ) / vx.shape[0] * 100
            if acc > best_val:  # the forward pass is already hard, so this is the circuit's acc
                best_val = acc
                best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
            print(f"  epoch {ep + 1:3d}/{c['epochs']}  loss {loss.item():.3f}  "
                  f"val {acc:.2f}%  (best {best_val:.2f}%)", flush=True)

        m.load_state_dict(best_state)

    @torch.no_grad()
    def predict(self, pix: np.ndarray) -> np.ndarray:
        m = self.model
        dev = next(m.parameters()).device
        x = _t(pix, dev)
        out = [m(x[i : i + 2048]).argmax(1).cpu() for i in range(0, len(x), 2048)]
        return torch.cat(out).numpy()  # ties -> lowest class, same as the emitted argmax

    def emit_verilog(self) -> str:
        m = self.model
        layers = [
            (lay.idx_a.cpu(), lay.idx_b.cpu(), lay.truth_table().cpu()) for lay in m.layers
        ]
        return emit_lutnet(m.thresholds, layers)


def build(**point) -> Submission:
    return BackpropLut(**point)
