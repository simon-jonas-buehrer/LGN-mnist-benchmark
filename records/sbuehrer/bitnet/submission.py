"""sbuehrer/bitnet: a ternary-weight, binary-activation MLP, translated fully into gates.

Weights are ternary (-1, 0, +1) as in BitNet; activations are single bits. A neuron is therefore
just a ternary-weighted count of its input bits, thresholded:

    h_j = [ (sum of inputs with weight +1) - (sum of inputs with weight -1) + b_j  >  0 ]

which in silicon is two popcounts, a subtract and a comparator -- an adder tree, not a lookup
table. That is the whole point of putting it on this benchmark: a ternary MLP and a logic-gate net
land on the same gate-equivalent axis, so you can see what dense ternary arithmetic costs in area
next to a learned gate net.

Training is straight-through: latent real weights are ternarized (TWN threshold) and the bias is
rounded to an integer on the forward pass, with gradients passed straight through, so the numbers
the trainer sees are exactly the integers the circuit computes. The forward pass is integer-exact,
so predict() matches the synthesized netlist bit for bit.

The encoder is a thermometer and the head is the usual group popcount + argmax, shared with the
other records, so only the hidden layers differ.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from mnistbench.data import Mnist, N_CLASSES, N_PIXELS, PIXEL_BITS
from mnistbench.hw import emit_popcount_argmax, emit_thermometer, even_thresholds
from mnistbench.spec import Submission

TITLE = "bitnet (ternary weights, binary activations, dense)"

# `bits` thermometer bits per pixel, `hidden` the ternary hidden widths, `readout` the final
# ternary layer (divisible by 10). `epochs` is a ceiling; validation early-stopping decides where
# each point stops. Each shape is the best one measured for its gate budget (see README).
POINTS = [
    {"name": "xs", "bits": 1, "hidden": (64,), "readout": 320, "epochs": 60},
    {"name": "s", "bits": 1, "hidden": (128,), "readout": 320, "epochs": 60},
    {"name": "m", "bits": 1, "hidden": (256, 256), "readout": 320, "epochs": 60},
    {"name": "l", "bits": 1, "hidden": (512,), "readout": 640, "epochs": 60},
    {"name": "xl", "bits": 3, "hidden": (512, 512), "readout": 640, "epochs": 60},
]


def _t(a: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).to(device)


def _ternarize(W: torch.Tensor) -> torch.Tensor:
    """TWN ternarization with a straight-through estimator: forward is {-1,0,1}, backward is identity.
    Per output neuron, delta = 0.7 * mean(|w|); |w| below delta -> 0, else sign(w)."""
    delta = 0.7 * W.abs().mean(0, keepdim=True)
    hard = torch.where(W > delta, 1.0, torch.where(W < -delta, -1.0, 0.0))
    return W + (hard - W).detach()


def _round_ste(b: torch.Tensor) -> torch.Tensor:
    return b + (b.round() - b).detach()


class TernaryLayer(torch.nn.Module):
    def __init__(self, n_in: int, n_out: int, g: torch.Generator) -> None:
        super().__init__()
        self.n_in = n_in
        self.W = torch.nn.Parameter(torch.randn(n_in, n_out, generator=g) * (1.0 / n_in ** 0.5))
        self.b = torch.nn.Parameter(torch.zeros(n_out))
        self.scale = n_in ** 0.5  # only shapes the STE gradient, not the (integer) forward value

    def preact(self, x: torch.Tensor) -> torch.Tensor:
        """Integer pre-activation, exactly what the circuit computes: x @ Wt + round(b)."""
        return x @ _ternarize(self.W) + _round_ste(self.b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.preact(x)
        hard = (pre > 0).float()
        soft = torch.sigmoid(pre / self.scale)  # smooth surrogate for the gradient only
        return hard + (soft - soft.detach())

    def wires(self) -> tuple[np.ndarray, np.ndarray]:
        Wt = _ternarize(self.W).detach().cpu().numpy().astype(np.int8)  # (n_in, n_out)
        B = _round_ste(self.b).detach().cpu().numpy().astype(np.int64)  # (n_out,)
        return Wt, B


class TernaryNet(torch.nn.Module):
    def __init__(self, bits: int, hidden: tuple[int, ...], readout: int, seed: int) -> None:
        super().__init__()
        if readout % N_CLASSES:
            raise ValueError(f"readout {readout} must be divisible by {N_CLASSES}")
        self.bits = bits
        self.thresholds = even_thresholds(bits)
        n_in = N_PIXELS * bits
        g = torch.Generator().manual_seed(seed)
        widths = list(hidden) + [readout]
        self.layers = torch.nn.ModuleList()
        for w in widths:
            self.layers.append(TernaryLayer(n_in, w, g))
            n_in = w
        self.readout = readout

    def encode(self, pix: torch.Tensor) -> torch.Tensor:
        t = torch.tensor(self.thresholds, device=pix.device, dtype=torch.int16)
        bits = pix.to(torch.int16).unsqueeze(-1) > t  # (N, 784, bits) pixel-major
        return bits.reshape(pix.shape[0], -1).float()

    def forward(self, pix: torch.Tensor) -> torch.Tensor:
        x = self.encode(pix)
        for lay in self.layers:
            x = lay(x)
        groups = x.reshape(x.shape[0], N_CLASSES, -1)
        return groups.sum(-1) / (self.readout // N_CLASSES) ** 0.5  # logits; argmax unaffected


class BitNet(Submission):
    def __init__(self, bits: int, hidden: tuple[int, ...], readout: int, epochs: int,
                 lr: float = 0.01, batch: int = 128, patience: int = 20) -> None:
        self.cfg = dict(bits=bits, hidden=tuple(hidden), readout=readout, epochs=epochs,
                        lr=lr, batch=batch, patience=patience)
        self.model: TernaryNet | None = None

    def train(self, data: Mnist, *, device: str = "cpu", seed: int = 0) -> None:
        c = self.cfg
        m = TernaryNet(c["bits"], c["hidden"], c["readout"], seed).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=c["lr"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=c["epochs"])
        x = _t(data.train_x, device)
        y = _t(data.train_y, device).long()
        vx, vy = _t(data.val_x, device), _t(data.val_y, device).long()
        n = x.shape[0]
        gen = torch.Generator(device=device).manual_seed(seed)

        best_val, best_state, best_ep = -1.0, None, 0
        for ep in range(c["epochs"]):
            perm = torch.randperm(n, generator=gen, device=device)
            for i in range(0, n, c["batch"]):
                idx = perm[i : i + c["batch"]]
                loss = F.cross_entropy(m(x[idx]), y[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sched.step()
            with torch.no_grad():
                acc = sum((m(vx[j : j + 4096]).argmax(1) == vy[j : j + 4096]).sum().item()
                          for j in range(0, vx.shape[0], 4096)) / vx.shape[0] * 100
            if acc > best_val:
                best_val, best_ep = acc, ep
                best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
            if ep % 5 == 0 or ep == c["epochs"] - 1:
                print(f"  epoch {ep + 1:3d}/{c['epochs']}  loss {loss.item():.3f}  "
                      f"val {acc:.2f}%  (best {best_val:.2f}% @ {best_ep + 1})", flush=True)
            if ep - best_ep >= c["patience"]:
                print(f"  early stop at epoch {ep + 1}: no gain since {best_ep + 1}", flush=True)
                break
        m.load_state_dict(best_state)
        self.model = m

    @torch.no_grad()
    def _logits(self, pix: np.ndarray) -> torch.Tensor:
        m = self.model
        dev = next(m.parameters()).device
        x = _t(pix, dev)
        out = [m(x[i : i + 4096]) for i in range(0, len(x), 4096)]
        return torch.cat(out)

    def predict(self, pix: np.ndarray) -> np.ndarray:
        return self._logits(pix).argmax(1).cpu().numpy()

    def scores(self, pix: np.ndarray) -> np.ndarray:
        # per-class firing fraction of the readout layer, in [0, 1]
        m = self.model
        dev = next(m.parameters()).device
        x = _t(pix, dev)
        out = []
        for i in range(0, len(x), 4096):
            h = m.encode(x[i : i + 4096])
            for lay in m.layers:
                h = (lay.preact(h) > 0).float()
            out.append(h.reshape(h.shape[0], N_CLASSES, -1).mean(-1).cpu())
        return torch.cat(out).numpy()

    def emit_verilog(self) -> str:
        return _emit(self.model)


def _emit(m: TernaryNet) -> str:
    enc, n_in = emit_thermometer(m.thresholds, sig="e")
    body = [enc]
    prev = [f"e[{i}]" for i in range(n_in)]  # names of the previous layer's bits
    for li, lay in enumerate(m.layers):
        Wt, B = lay.wires()                  # (n_in, n_out), (n_out,)
        n_out = Wt.shape[1]
        cw = max(1, int(Wt.shape[0]).bit_length())  # bits to hold a popcount over the inputs
        pw = cw + 2                                  # signed pre-activation width
        body.append(f"  // ternary layer {li}: {Wt.shape[0]} -> {n_out} (ternary weights)")
        body.append(f"  wire [{n_out - 1}:0] h{li};")
        for j in range(n_out):
            pos = [prev[i] for i in np.nonzero(Wt[:, j] == 1)[0]]
            neg = [prev[i] for i in np.nonzero(Wt[:, j] == -1)[0]]
            ps = " + ".join(pos) if pos else "0"
            ns = " + ".join(neg) if neg else "0"
            # zero-extend the two popcounts to signed, subtract, add the integer bias, test > 0.
            # h = [ (#inputs with weight +1 that are high) - (#with weight -1) + b_j  >  0 ]
            body.append(f"  wire [{cw - 1}:0] p{li}_{j} = {ps};")
            body.append(f"  wire [{cw - 1}:0] n{li}_{j} = {ns};")
            body.append(f"  wire signed [{pw}:0] a{li}_{j} = "
                        f"$signed({{2'b0, p{li}_{j}}}) - $signed({{2'b0, n{li}_{j}}}) "
                        f"+ ({int(B[j])});")
            body.append(f"  assign h{li}[{j}] = a{li}_{j} > 0;")
        prev = [f"h{li}[{j}]" for j in range(n_out)]

    head = emit_popcount_argmax(prev, N_CLASSES)
    return f"""// generated by sbuehrer/bitnet -- {len(m.layers)} ternary layers
module top (input [{N_PIXELS * PIXEL_BITS - 1}:0] pix, output logic [3:0] cls);
  wire [{n_in - 1}:0] e;

{chr(10).join(body)}

{head}
endmodule
"""


def build(**point) -> Submission:
    return BitNet(**point)
