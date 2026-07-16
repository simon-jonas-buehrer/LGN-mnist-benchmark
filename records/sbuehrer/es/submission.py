"""sbuehrer/es: fixed Monarch wiring, gate truth tables learned by evolution strategies.

Like the genetic record this is gradient-free and perturbation-based, but instead of a hard
mutation hill-climb it keeps a real-valued state and estimates a gradient from the perturbations.

Every gate has a 4-entry truth table. Each entry is a logit theta; p = sigmoid(theta) is read as
the probability that entry is 1. Each generation:

  1. draw N rollouts, each a full truth table sampled bit ~ Bernoulli(p);
  2. score every rollout by margin (votes for the true class minus the best wrong class) on the
     same minibatch;
  3. turn the scores into advantages with a leave-one-out baseline (each rollout compared to the
     mean of the others), std-normalized across the N rollouts;
  4. REINFORCE update, theta += lr/N * sum_i adv_i * (bits_i - p). Entries that were 1 in the
     better-than-average rollouts get pushed up, and vice versa.

The default update is elite selection (cross-entropy method / PBIL): keep the top rollouts and move
each probability toward their mean bit. It is far lower variance than a REINFORCE gradient over
~16k logits from one scalar reward, which is why it works where plain score-function ES stalls
(REINFORCE is kept as an option). The wiring is a fixed butterfly (gate j reads j and j ^ (1<<k),
stride halving each layer), so the receptive field reaches every pixel in log-depth and only the
tables are learned. The emitted circuit uses the greedy table (p > 0.5), exactly what predict()
runs, so the circuit matches predict() bit for bit.

Init is random (not residual): a sampled ES has nothing to select between if it starts at identity.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from mnistbench.data import Mnist, N_CLASSES, N_PIXELS
from mnistbench.hw import emit_lutnet, even_thresholds
from mnistbench.spec import Submission

TITLE = "es (fixed butterfly wiring, evolution strategies on gate tables)"

# depth = log2(width) + 1: the shallowest butterfly in which every readout gate sees every pixel.
POINTS = [
    {"name": "xs", "bits": 1, "width": 1024, "depth": 11, "readout": 320},
    {"name": "s", "bits": 1, "width": 2048, "depth": 12, "readout": 640},
    {"name": "m", "bits": 3, "width": 4096, "depth": 13, "readout": 640},
    {"name": "l", "bits": 3, "width": 8192, "depth": 14, "readout": 1280},
]


def _t(a: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).to(device)


def _log2(n: int) -> int:
    if n & (n - 1):
        raise ValueError(f"{n} is not a power of two")
    return n.bit_length() - 1


def _butterfly_src(in_dim: int, out_dim: int, stage: int) -> torch.Tensor:
    """(2, out_dim) local source indices into `in_dim`, fan-in 2, deterministic butterfly tap.

    Gate j reads j and j ^ (1 << k), with the stride k halving every layer (k cycles over
    0 .. log2(width)-1), so the receptive field doubles per layer: after log2(width) body layers
    every gate depends on every pixel. (An earlier Monarch tap used a constant across-group stride
    g/2 which is a 2-cycle, so the receptive field never grew with depth and the net sat at chance.
    Verify with receptive_field(), never by eye.)
    """
    j = torch.arange(out_dim)
    if in_dim == out_dim:                       # body: the butterfly proper
        k = stage % _log2(in_dim)
        return torch.stack([j, j ^ (1 << k)])
    if out_dim > in_dim:                        # encoder -> first layer: cover every input bit
        return torch.stack([(2 * j) % in_dim, (2 * j + 1) % in_dim])
    a = (j * in_dim) // out_dim                 # readout: spread the tap over the last body layer
    return torch.stack([a, (a + in_dim // 2) % in_dim])


def receptive_field(net: "MonarchES") -> np.ndarray:
    """(readout_width,) how many encoder bits each readout gate actually depends on."""
    reach = np.eye(net.n_in, dtype=bool)
    for l, s in enumerate(net.srcs):
        base = 0 if l == 0 else net.offs[l - 1]
        reach = reach[s[0].cpu().numpy() - base] | reach[s[1].cpu().numpy() - base]
    return reach.sum(1)


def _encode(pix: torch.Tensor, thresholds) -> torch.Tensor:
    thr = torch.tensor(thresholds, device=pix.device, dtype=torch.int16)
    bits = pix.to(torch.int16).unsqueeze(-1) > thr
    return bits.reshape(pix.shape[0], -1).T.contiguous().to(torch.uint8)


class MonarchES:
    """Fixed Monarch fan-in-2 wiring; per-gate table logits theta (learned by ES)."""

    def __init__(self, bits: int, width: int, depth: int, readout: int, device: str,
                 g: torch.Generator) -> None:
        if readout % N_CLASSES:
            raise ValueError(f"readout {readout} must be divisible by {N_CLASSES}")
        _log2(width)  # the butterfly needs a power-of-two body; fail loudly, not silently
        self.thresholds = even_thresholds(bits)
        self.n_in = N_PIXELS * bits
        self.device = device
        self.widths = [width] * depth + [readout]

        self.offs = [self.n_in]
        self.in_base = [0]
        in_dim, in_base = self.n_in, 0
        self.srcs: list[torch.Tensor] = []
        for l, w in enumerate(self.widths):
            # stage l-1: the encoder layer is not a butterfly stage, so the cycle starts after it
            mon = _butterfly_src(in_dim, w, l - 1).contiguous()  # (2, w)
            self.srcs.append((mon + in_base).to(device))
            in_base = self.offs[-1]
            self.in_base.append(in_base)
            self.offs.append(self.offs[-1] + w)
            in_dim = w

        # probability state per table entry, near 0.5 so sampling explores both bits (random init;
        # a residual/identity start would give the search nothing to select between)
        self.p = [(0.5 + 0.02 * torch.randn(w, 4, generator=g, device=device)).clamp(0.05, 0.95)
                  for w in self.widths]

    @property
    def n_sig(self) -> int:
        return self.offs[-1]

    def forward(self, enc: torch.Tensor, tts: list[torch.Tensor]) -> torch.Tensor:
        """enc (n_in, B) uint8, tts list of (w,4) uint8 -> votes (B, 10)."""
        acts = torch.zeros((self.n_sig, enc.shape[1]), dtype=torch.uint8, device=enc.device)
        acts[: self.n_in] = enc
        for l, s in enumerate(self.srcs):
            p = (acts[s[0]].long() << 1) | acts[s[1]].long()      # (w, B)
            acts[self.offs[l] : self.offs[l + 1]] = tts[l].gather(1, p)
        out = acts[self.offs[-2] : self.offs[-1]]                 # (R, B)
        return out.reshape(N_CLASSES, -1, out.shape[1]).sum(1).T.float()  # (B, 10)

    def greedy_tts(self) -> list[torch.Tensor]:
        return [(p > 0.5).to(torch.uint8) for p in self.p]

    def tt_packed(self) -> list[torch.Tensor]:
        out = []
        for t in self.greedy_tts():
            out.append((t[:, 0] | (t[:, 1] << 1) | (t[:, 2] << 2) | (t[:, 3] << 3)).cpu())
        return out


def _margin(votes: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    true = votes.gather(1, y[:, None]).squeeze(1)
    wrong = votes.scatter(1, y[:, None], -1e9).max(1).values
    return true - wrong


class EvoStrat(Submission):
    def __init__(self, bits: int, width: int, depth: int, readout: int, gens: int = 20000,
                 rollouts: int = 32, batch: int = 8192, lr: float = 0.2, rule: str = "cem",
                 elite: float = 0.25, patience: int = 20, eval_every: int = 200) -> None:
        self.cfg = dict(bits=bits, width=width, depth=depth, readout=readout, gens=gens,
                        rollouts=rollouts, batch=batch, lr=lr, rule=rule, elite=elite,
                        patience=patience, eval_every=eval_every)
        self.net: MonarchES | None = None

    def train(self, data: Mnist, *, device: str = "cpu", seed: int = 0) -> None:
        c = self.cfg
        g = torch.Generator(device=device).manual_seed(seed)
        net = MonarchES(c["bits"], c["width"], c["depth"], c["readout"], device, g)

        enc_tr = _encode(_t(data.train_x, device), net.thresholds)
        y_tr = _t(data.train_y, device).long()
        enc_va = _encode(_t(data.val_x, device), net.thresholds)
        y_va = _t(data.val_y, device).long()
        N = c["rollouts"]

        n_elite = max(1, int(round(N * c["elite"])))
        best_val, best_p, stale = -1.0, [p.clone() for p in net.p], 0
        t0 = time.time()
        for it in range(c["gens"]):
            idx = torch.randint(enc_tr.shape[1], (c["batch"],), generator=g, device=device)
            xb, yb = enc_tr[:, idx], y_tr[idx]

            rolls, R = [], []
            for _ in range(N):
                bits = [(torch.rand(p.shape, generator=g, device=device) < p) for p in net.p]
                R.append(_margin(net.forward(xb, [b.to(torch.uint8) for b in bits]), yb).mean().item())
                rolls.append(bits)
            Rt = torch.tensor(R, device=device)

            if c["rule"] == "cem":
                # move each probability toward the mean bit of the top-e rollouts (elite selection)
                elite = torch.topk(Rt, n_elite).indices.tolist()
                for l in range(len(net.p)):
                    em = torch.stack([rolls[i][l].float() for i in elite]).mean(0)
                    net.p[l] = ((1 - c["lr"]) * net.p[l] + c["lr"] * em).clamp(0.02, 0.98)
            else:
                # REINFORCE with leave-one-out baseline (higher variance)
                adv = Rt - (Rt.sum() - Rt) / (N - 1)
                adv = adv / (adv.std() + 1e-8)
                for i, bits in enumerate(rolls):
                    for l in range(len(net.p)):
                        net.p[l] = (net.p[l] + c["lr"] / N * adv[i].item()
                                    * (bits[l].float() - net.p[l])).clamp(0.02, 0.98)

            if (it + 1) % c["eval_every"] == 0 or it + 1 == c["gens"]:
                acc = self._accuracy(net, enc_va, y_va)
                if acc > best_val:
                    best_val, best_p, stale = acc, [p.clone() for p in net.p], 0
                else:
                    stale += 1
                print(f"  gen {it + 1:6d}/{c['gens']}  val {acc:.2f}%  "
                      f"(best {best_val:.2f}%, stale {stale})  "
                      f"{(it + 1) / (time.time() - t0):.1f} gen/s", flush=True)
                if stale >= c["patience"]:
                    print(f"  early stop at gen {it + 1}: converged (best {best_val:.2f}%)",
                          flush=True)
                    break
        for p, bp in zip(net.p, best_p):
            p.copy_(bp)
        self.net = net

    @torch.no_grad()
    def _accuracy(self, net, enc, y, chunk: int = 4096):
        tts = net.greedy_tts()
        right = 0
        for i in range(0, enc.shape[1], chunk):
            v = net.forward(enc[:, i : i + chunk], tts)
            right += (v.argmax(1) == y[i : i + chunk]).sum().item()
        return right / enc.shape[1] * 100

    @torch.no_grad()
    def predict(self, pix: np.ndarray) -> np.ndarray:
        net = self.net
        tts = net.greedy_tts()
        enc = _encode(_t(pix, net.device), net.thresholds)
        out = []
        for i in range(0, enc.shape[1], 4096):
            out.append(net.forward(enc[:, i : i + 4096], tts).argmax(1).cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def scores(self, pix: np.ndarray) -> np.ndarray:
        net = self.net
        gsz = net.widths[-1] // N_CLASSES
        tts = net.greedy_tts()
        enc = _encode(_t(pix, net.device), net.thresholds)
        out = []
        for i in range(0, enc.shape[1], 4096):
            out.append((net.forward(enc[:, i : i + 4096], tts) / gsz).cpu())
        return torch.cat(out).numpy()

    def emit_verilog(self) -> str:
        net = self.net
        layers = [(s[0].cpu(), s[1].cpu(), tt) for s, tt in zip(net.srcs, net.tt_packed())]
        return emit_lutnet(net.thresholds, layers)


def build(**point) -> Submission:
    return EvoStrat(**point)
