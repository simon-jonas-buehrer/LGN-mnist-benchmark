"""sbuehrer/dfa: fixed butterfly wiring, truth tables learned by direct feedback alignment.

Nothing in this record is learned except the 4 bits of each gate's truth table. Both of the
network's structures are fixed and non-learnable:

  Forward. A butterfly (FFT) pattern wires every gate to exactly two signals of the layer below:
  gate j reads j and j ^ (1 << k), with the stride k halving every layer. It is deterministic and
  never touched by the optimizer. The stride cycle makes the receptive field genuinely double per
  layer, so after log2(width) layers every gate sees every pixel -- but see DEPTH below: it turns
  out you do not WANT that. The nets here are 5 layers deep and each gate sees ~64 of 784 pixels.
  The cycle still matters, because a butterfly with a constant stride mixes nothing at all.

  Backward. A fixed random matrix B_l projects the output error DIRECTLY onto layer l. There is no
  backward sweep: no error signal ever crosses a layer boundary, and no chain rule is ever applied
  between layers. This is direct feedback alignment (Nokland 2016, arXiv:1609.01596).

Why DFA fits a LUT net especially well. The forward pass is already exact bits, so a gate's output
is just T[p] where p = 2a+b is the pattern its two inputs happen to present. The derivative of the
output with respect to the gate's own table is therefore the *indicator of the active pattern*:
only T[p] moves, the other three entries get nothing. So the whole per-layer update is one
scatter_add, and it needs exactly two things -- the layer's own forward patterns, and the broadcast
error:

    e       = softmax(votes/tau) - onehot(y)         # (B,10), the only global signal there is
    delta_l = e @ B_l                                # (B,w), B_l fixed random (10,w)
    G[i,p]  = sum_b delta_l[b,i] * 1[p_bi == p]      # scatter_add, (w,4)
    z.grad  = G * 0.5*cos(z)                         # chain THROUGH a gate, never ACROSS one

The layers are updated in any order, or in parallel; they do not talk. The readout layer uses its
own local gradient (e[b, class(i)] / tau), which is DFA's standard convention for the output layer
and is still local. `.backward()` is never called and no autograd graph is ever built -- the whole
step runs under torch.no_grad(), which is what makes the no-backprop claim structural rather than
merely intended. Adam and the cosine schedule are stock; only the gradient is hand-written.

B_l is dense random and is a *training-time* object. It is never synthesized, so it costs zero gate
equivalents; the structure that becomes silicon is the butterfly wiring.

Latents are sin-binarized, hard = 1[sin(z) > 0], borrowed from the backprop record: sin is
periodic, so a latent never saturates and there is always a slope toward the nearest 0/1 basin.
Init is residual (every gate passes input A, tt 0b1100) so the net starts as an identity path and a
deep gate's change actually reaches the readout. The init sits at +-pi/4 rather than +-pi/2: pi/2
is the exactly-right table but cos(pi/2) = 0, so the gradient would be identically zero and the net
would never move at all.

The forward pass is exact boolean, so the val accuracy printed during training IS the circuit's
accuracy, and the harness's 512-image model-vs-netlist check is a formality rather than a hazard.

The shape the sweeps settled on (see README): 5 layers, bits=1, and everything else in the readout.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from mnistbench.data import Mnist, N_CLASSES, N_PIXELS
from mnistbench.hw import emit_lutnet, even_thresholds
from mnistbench.spec import Submission

TITLE = "dfa (fixed butterfly wiring, direct feedback alignment)"

# Every point is the winner of a measured sweep at its size, not a guess (see the docstring). The
# shape rule that fell out: 5 layers, bits 1, and spend everything else on the READOUT.
#
# The knob is `layers`, NOT `depth`: bench.merge_record merges the MEASURED fields over this dict,
# and one of them is "depth" (the synthesized netlist's longest-path level count). A POINTS key
# named `depth` is silently overwritten -- results.json then reports depth 192 for a 5-layer net,
# and nobody can rebuild the record from its own results. Never name a POINTS key after a measured
# field: ge, area_um2, cells, nand, inv, depth, test_acc, val_acc, train_s, device, seed, test_ce,
# ce_temp.
POINTS = [
    {"name": "xs", "bits": 1, "width": 256, "layers": 5, "readout": 640, "epochs": 60},
    {"name": "s", "bits": 1, "width": 512, "layers": 5, "readout": 1280, "epochs": 60},
    {"name": "m", "bits": 1, "width": 1024, "layers": 5, "readout": 5120, "epochs": 60},
    {"name": "l", "bits": 1, "width": 2048, "layers": 5, "readout": 10240, "epochs": 60},
    {"name": "xl", "bits": 1, "width": 4096, "layers": 5, "readout": 20480, "epochs": 60},
]


def _t(a: np.ndarray, device: str) -> torch.Tensor:
    """The harness speaks numpy; torch starts here."""
    return torch.from_numpy(np.ascontiguousarray(a)).to(device)


# ---- fixed structure: the butterfly tap --------------------------------------------------------


def _log2(n: int) -> int:
    if n & (n - 1):
        raise ValueError(f"{n} is not a power of two")
    return n.bit_length() - 1


def _butterfly_src(in_dim: int, out_dim: int, stage: int) -> torch.Tensor:
    """(2, out_dim) local source indices into `in_dim`. Fan-in 2, deterministic, non-learnable.

    The body layers are the real structure: gate j reads j and j ^ (1 << k), with the stride k
    HALVING every layer (k cycles 0, 1, 2, ... log2(width)-1). That is the FFT butterfly, and it is
    what makes the receptive field actually double per layer: after log2(width) body layers every
    gate depends on every signal.

    The stride must vary. A butterfly whose stride is CONSTANT at width/2 pairs j with j^(w/2)
    forever, and applying that twice returns to j -- a 2-cycle, so the receptive field saturates
    after one layer and depth buys nothing. (That is the bug in the Monarch tap this record started
    from; see the README. It is invisible in the wiring and only shows up if you actually trace
    reachability, so no accuracy number names it.)
    """
    j = torch.arange(out_dim)
    if in_dim == out_dim:                       # body: the butterfly proper
        k = stage % _log2(in_dim)
        return torch.stack([j, j ^ (1 << k)])
    if out_dim > in_dim:                        # encoder -> first layer: cover every input bit
        return torch.stack([(2 * j) % in_dim, (2 * j + 1) % in_dim])
    a = (j * in_dim) // out_dim                 # readout: spread the tap over the last body layer
    return torch.stack([a, (a + in_dim // 2) % in_dim])


def hard_bit(z: torch.Tensor) -> torch.Tensor:
    """1[sin(z) > 0] -- periodic, so a latent never saturates out of reach of the gradient."""
    return (torch.sin(z) > 0).to(torch.uint8)


# Residual init: T[p] = (p>>1)&1 = a, i.e. every gate passes input A (tt 0b1100). Sign per entry;
# magnitude is pi/4, NOT pi/2 -- at pi/2 the table is exactly right but cos(pi/2)=0 kills the
# gradient and nothing would ever move.
_RES_SIGN = torch.tensor([-1.0, -1.0, 1.0, 1.0])

# Symmetry break on top of the residual init: identical latents would get identical updates and the
# layer would collapse to one gate.
_JITTER = 0.1


class ButterflyNet:
    """Fixed butterfly fan-in-2 wiring; per-gate 4-entry sin latent (learned); fixed random B."""

    def __init__(self, bits: int, width: int, layers: int, readout: int, device: str,
                 g: torch.Generator) -> None:
        if readout % N_CLASSES:
            raise ValueError(f"readout {readout} must be divisible by {N_CLASSES}")
        _log2(width)  # the butterfly needs a power-of-two body; fail loudly, not silently
        self.bits = bits
        self.thresholds = even_thresholds(bits)
        self.n_in = N_PIXELS * bits
        self.device = device
        self.widths = [width] * layers + [readout]
        self.tau = (readout // N_CLASSES) ** 0.5  # keep logits sane, as in the backprop record

        # fixed wiring: srcs[l] = (2, w) GLOBAL ids; layer l reads only layer l-1 (or the encoder)
        self.offs = [self.n_in]
        in_dim, in_base = self.n_in, 0
        self.srcs: list[torch.Tensor] = []
        for l, w in enumerate(self.widths):
            # stage l-1: the encoder layer is not a butterfly stage, so the cycle starts after it
            bf = _butterfly_src(in_dim, w, l - 1).contiguous()  # (2, w) local into in_dim
            self.srcs.append((bf + in_base).to(device))
            in_base = self.offs[-1]                # next layer reads THIS layer's outputs
            self.offs.append(self.offs[-1] + w)
            in_dim = w

        # learned: the only parameters in the whole record
        self.z = [
            (_RES_SIGN.to(device) * (torch.pi / 4)).expand(w, 4).contiguous()
            + torch.randn(w, 4, generator=g, device=device) * _JITTER
            for w in self.widths
        ]
        # fixed random feedback: the backward "model". Never learned, never synthesized.
        self.B = [
            torch.randn(N_CLASSES, w, generator=g, device=device) / N_CLASSES**0.5
            for w in self.widths[:-1]
        ]

    @property
    def n_sig(self) -> int:
        return self.offs[-1]

    def tables(self) -> list[torch.Tensor]:
        """(w,4) uint8 hard truth tables -- what the forward pass and the Verilog both use."""
        return [hard_bit(z) for z in self.z]

    def forward(self, enc: torch.Tensor, T: list[torch.Tensor] | None = None) -> torch.Tensor:
        """enc (n_in, B) uint8 -> acts (n_sig, B) uint8. Exact bits; no relaxation anywhere."""
        T = self.tables() if T is None else T
        acts = torch.zeros((self.n_sig, enc.shape[1]), dtype=torch.uint8, device=enc.device)
        acts[: self.n_in] = enc
        for l, s in enumerate(self.srcs):
            p = (acts[s[0]].long() << 1) | acts[s[1]].long()   # (w, B) in {0,1,2,3}
            acts[self.offs[l] : self.offs[l + 1]] = T[l].gather(1, p)
        return acts

    def votes(self, acts: torch.Tensor) -> torch.Tensor:
        out = acts[self.offs[-2] : self.offs[-1]]  # (R, B)
        return out.reshape(N_CLASSES, -1, out.shape[1]).sum(1).T.float()  # (B, 10)

    def tt(self) -> list[torch.Tensor]:
        """pack each (w,4) table into a (w,) 4-bit int: bit p = T[p], the encoding hw wants."""
        return [(t[:, 0] | (t[:, 1] << 1) | (t[:, 2] << 2) | (t[:, 3] << 3)).cpu()
                for t in self.tables()]


def _encode(pix: torch.Tensor, net: ButterflyNet) -> torch.Tensor:
    """(N,784) uint8 -> (n_in, N) uint8, pixel-major, matching hw.emit_thermometer."""
    thr = torch.tensor(net.thresholds, device=pix.device, dtype=torch.int16)
    bits = pix.to(torch.int16).unsqueeze(-1) > thr  # (N, 784, bits)
    return bits.reshape(pix.shape[0], -1).T.contiguous().to(torch.uint8)


class DfaLut(Submission):
    def __init__(self, bits: int, width: int, layers: int, readout: int, epochs: int,
                 lr: float = 0.01, batch: int = 256, patience: int = 12) -> None:
        self.cfg = dict(bits=bits, width=width, layers=layers, readout=readout, epochs=epochs,
                        lr=lr, batch=batch, patience=patience)
        self.net: ButterflyNet | None = None

    @torch.no_grad()
    def train(self, data: Mnist, *, device: str = "cpu", seed: int = 0) -> None:
        c = self.cfg
        torch.manual_seed(seed)
        g = torch.Generator(device=device).manual_seed(seed)
        net = ButterflyNet(c["bits"], c["width"], c["layers"], c["readout"], device, g)

        enc_tr = _encode(_t(data.train_x, device), net)  # (n_in, N)
        y_tr = _t(data.train_y, device).long()
        enc_va = _encode(_t(data.val_x, device), net)
        y_va = _t(data.val_y, device).long()

        # Adam only ever sees gradients we computed by hand; it never runs a backward pass.
        params = [torch.nn.Parameter(z) for z in net.z]
        net.z = params
        opt = torch.optim.Adam(params, lr=c["lr"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=c["epochs"])

        n = enc_tr.shape[1]
        steps = n // c["batch"]
        best_val, best_z, best_ep, t0 = -1.0, [z.detach().clone() for z in net.z], 0, time.time()
        for ep in range(c["epochs"]):
            perm = torch.randperm(n, generator=g, device=device)
            for i in range(steps):
                idx = perm[i * c["batch"] : (i + 1) * c["batch"]]
                loss = self._step(net, enc_tr[:, idx], y_tr[idx], opt)
            sched.step()

            acc = self._accuracy(net, enc_va, y_va)
            if acc > best_val:
                best_val, best_ep = acc, ep
                best_z = [z.detach().clone() for z in net.z]
            print(f"  epoch {ep + 1:3d}/{c['epochs']}  loss {loss:.3f}  val {acc:.2f}%  "
                  f"(best {best_val:.2f}% @ {best_ep + 1})  "
                  f"{(ep + 1) / (time.time() - t0):.2f} ep/s", flush=True)
            if ep - best_ep >= c["patience"]:
                print(f"  early stop at epoch {ep + 1}: no gain since {best_ep + 1}", flush=True)
                break

        for z, bz in zip(net.z, best_z):
            z.copy_(bz)
        self.net = net

    # ---- the DFA update ------------------------------------------------------------------------
    def _step(self, net: ButterflyNet, enc: torch.Tensor, y: torch.Tensor, opt) -> float:
        """One direct-feedback-alignment step. No graph, no .backward(), no cross-layer signal."""
        B = enc.shape[1]
        T = net.tables()
        acts = net.forward(enc, T)

        # the ONLY global quantity: the output error, broadcast from here to every layer at once
        logits = net.votes(acts) / net.tau                      # (B,10)
        prob = torch.softmax(logits, 1)
        loss = -torch.log(prob[torch.arange(B, device=enc.device), y] + 1e-12).mean()
        e = prob.clone()
        e[torch.arange(B, device=enc.device), y] -= 1.0
        e /= B                                                  # mean over the batch

        for l, w in enumerate(net.widths):
            s = net.srcs[l]
            p = (acts[s[0]].long() << 1) | acts[s[1]].long()    # (w,B) active pattern, recomputed
            if l == len(net.widths) - 1:
                # readout: its own local gradient. dlogit_c/d bit_i = 1/tau for i in group c.
                cls_of = torch.arange(w, device=enc.device) // (w // N_CLASSES)
                delta = e[:, cls_of] / net.tau                  # (B,w)
            else:
                delta = e @ net.B[l]                            # (B,w) -- the direct projection
            # only the ACTIVE table entry of each gate gets gradient: G[i,p] = sum_b delta[b,i]
            G = torch.zeros(w * 4, device=enc.device)
            idx = torch.arange(w, device=enc.device)[:, None] * 4 + p   # (w,B)
            G.scatter_add_(0, idx.reshape(-1), delta.t().reshape(-1))
            net.z[l].grad = G.view(w, 4) * (0.5 * torch.cos(net.z[l]))

        opt.step()
        return loss.item()

    # ---- eval ----------------------------------------------------------------------------------
    def _chunk(self) -> int:
        """Rows per forward pass: acts is (n_sig, rows) uint8, so cap that at ~256 MB."""
        c = self.cfg
        n_sig = N_PIXELS * c["bits"] + c["width"] * c["layers"] + c["readout"]
        return max(64, min(4096, 2**28 // n_sig))

    @torch.no_grad()
    def _accuracy(self, net: ButterflyNet, enc: torch.Tensor, y: torch.Tensor) -> float:
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
        return torch.cat(out).numpy()  # ties -> lowest class, same as the emitted argmax

    @torch.no_grad()
    def scores(self, pix: np.ndarray) -> np.ndarray:
        """Per-class firing fraction in [0,1]: of each class's readout gates, how many fired."""
        net = self.net
        gsz = net.widths[-1] // N_CLASSES
        enc = _encode(_t(pix, net.device), net)
        ch = self._chunk()
        out = [(net.votes(net.forward(enc[:, i : i + ch])) / gsz).cpu()
               for i in range(0, enc.shape[1], ch)]
        return torch.cat(out).numpy()

    def emit_verilog(self) -> str:
        net = self.net
        layers = [(s[0].cpu(), s[1].cpu(), tt) for s, tt in zip(net.srcs, net.tt())]
        return emit_lutnet(net.thresholds, layers)


def build(**point) -> Submission:
    return DfaLut(**point)
