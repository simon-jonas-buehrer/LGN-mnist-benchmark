"""sbuehrer/genetic -- learn the WIRING of a fixed NAND net by mutation hill-climbing.

The mirror image of sbuehrer/backprop. There, every gate could become any of the 16 boolean
functions but the wires were frozen; here every gate is a NAND, forever, and the only free
parameters are which two signals each gate reads. NAND alone is functionally complete, so this
search space contains every circuit the LUT net can express -- the question is whether a
gradient-free search can find anything in it.

The algorithm is deliberately the simplest thing that deserves the name:

    for each generation:
        make k-1 mutants of the current wiring (rewire `mut` gate endpoints at random)
        score all k (the incumbent is always one of them) on the SAME minibatch
        keep the best

Three details that are not cosmetic:

  * FITNESS IS A MARGIN, NOT ACCURACY. Accuracy on a minibatch changes only when a prediction
    flips, so almost every single-wire mutation scores identically and the search random-walks.
    The margin (votes for the true class minus the best wrong class) moves whenever any vote
    moves, which turns a flat plateau into a slope.
  * THE SELECTION BATCH MUST BE BIG. One rewired wire moves the margin by a hair, so a small
    batch drowns that signal in sampling noise and selection picks the luckier mutant instead
    of the better one. Measured on the `xs` point, 20k generations, everything else equal:
    batch 1024 -> 22.6%, batch 4096 -> 24.9%, **batch 8192 -> 60.4%**. This one number is worth
    more than any amount of cleverness elsewhere in the search. It does saturate, though -- on
    `m`: 8192 -> 78.7%, 16384 -> 79.6%, 32768 -> 79.6%. 16384 is the knee, and past it you are
    paying 2x the time per generation for nothing.
  * DELTA FORWARD. A mutant differs from the incumbent only from its lowest mutated layer
    upward, so everything below that is reused. Cheap, exact, and it is most of the speed.
  * GENERATIONS ARE THE SCARCE RESOURCE, so spend them on generations. `k` (mutants per
    generation) trades throughput for a better pick each step, and there is an interior optimum:
    on `m`, k=4 -> 82.3% (59 gen/s), **k=8 -> 83.3% (32 gen/s)**, k=16 -> 81.4% (10 gen/s, and
    still climbing when it ran out of time -- it is not worse, it is just slower per step, which
    is the same thing when the budget is wall-clock). Likewise mut=1 beats mut=2 and mut=4.

WHAT THE CAP WAS HIDING. This record used to stop at 20k-40k generations and report ~81%, and
the curve was read as "the hill-climber plateaus at 81% no matter how many gates you give it".
That was false, and it was an artifact of the cap: every point was still finding new bests when
the generation budget ran out. Given room to converge, `m` reaches 83.3%. The lesson is not that
the GA is secretly good -- it still loses badly to backprop per gate -- but that a stopping rule
you chose for convenience will happily masquerade as a property of the algorithm. Every point
here now runs until it stops improving (patience), never to a fixed generation count.

No gradients are involved anywhere. This is here to be beaten -- and to make visible what the
gate count of a hill-climbed circuit looks like next to a differentiated one.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from mnistbench.data import Mnist, N_CLASSES, N_PIXELS
from mnistbench.hw import emit_lutnet, even_thresholds
from mnistbench.spec import Submission


def _t(a: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).to(device)

TITLE = "genetic (learned wiring, all gates NAND)"

NAND_TT = 0b0111  # bit (2a+b) of ~(a & b): f(0,0)=f(0,1)=f(1,0)=1, f(1,1)=0

# `gens` is a CEILING, not a target. Training stops when validation has not improved for
# `patience` evaluations, so each point runs to its own convergence. The old caps (20k-40k) were
# the single biggest lie in this record: every point was still climbing when it hit them.
POINTS = [
    {"name": "xs", "bits": 1, "widths": (256, 256, 160), "gens": 1000000},
    {"name": "s", "bits": 1, "widths": (1024, 1024, 320), "gens": 1000000},
    {"name": "m", "bits": 3, "widths": (2048, 2048, 2048, 640), "gens": 1000000},
    {"name": "l", "bits": 3, "widths": (4096, 4096, 4096, 4096, 1280), "gens": 1000000},
]


class NandNet:
    """Wiring only: srcs[l] is a (2, width) tensor of signal ids, all gates NAND."""

    def __init__(self, bits: int, widths: tuple[int, ...], device: str, g: torch.Generator):
        if widths[-1] % N_CLASSES:
            raise ValueError(f"readout width {widths[-1]} must be divisible by {N_CLASSES}")
        self.bits = bits
        self.widths = widths
        self.thresholds = even_thresholds(bits)
        self.n_in = N_PIXELS * bits
        self.device = device

        self.offs = [self.n_in]
        self.srcs: list[torch.Tensor] = []
        for w in widths:
            off = self.offs[-1]
            # a gate reads any strictly earlier signal -> the graph is acyclic by construction
            self.srcs.append(torch.randint(off, (2, w), generator=g, device=device))
            self.offs.append(off + w)

    @property
    def n_sig(self) -> int:
        return self.offs[-1]

    def clone(self) -> "NandNet":
        new = object.__new__(NandNet)
        new.__dict__.update(self.__dict__)
        new.srcs = [s.clone() for s in self.srcs]
        return new

    def forward(self, enc: torch.Tensor, acts: torch.Tensor | None = None, start: int = 0):
        """enc is (n_in, B) bool. Returns the full (n_sig, B) activation buffer.

        With `acts` and `start`, only layers >= start are recomputed -- the delta forward.
        """
        if acts is None:
            acts = torch.zeros((self.n_sig, enc.shape[1]), dtype=torch.bool, device=enc.device)
            acts[: self.n_in] = enc
        for l in range(start, len(self.srcs)):
            s = self.srcs[l]
            acts[self.offs[l] : self.offs[l + 1]] = ~(acts[s[0]] & acts[s[1]])
        return acts

    def votes(self, acts: torch.Tensor) -> torch.Tensor:
        """(B, 10) class vote counts: the readout layer split into 10 contiguous groups."""
        out = acts[self.offs[-2] : self.offs[-1]]  # (R, B)
        return out.reshape(N_CLASSES, -1, out.shape[1]).sum(1).T.float()


def margin(votes: torch.Tensor, y: torch.Tensor) -> float:
    """votes for the true class minus the best wrong class, averaged. A slope, not a cliff."""
    true = votes.gather(1, y[:, None]).squeeze(1)
    wrong = votes.scatter(1, y[:, None], -1e9).max(1).values
    return (true - wrong).mean().item()


class GeneticNand(Submission):
    def __init__(self, bits: int, widths: tuple[int, ...], gens: int, k: int = 8,
                 mut: int = 1, batch: int = 16384, eval_every: int = 5000,
                 patience: int = 20) -> None:
        self.cfg = dict(bits=bits, widths=tuple(widths), gens=gens, k=k, mut=mut,
                        batch=batch, eval_every=eval_every, patience=patience)
        self.net: NandNet | None = None

    # ---- the search ----------------------------------------------------------------
    def _mutate(self, net: NandNet, g: torch.Generator) -> tuple[NandNet, int]:
        """Rewire `mut` random gate endpoints. Returns the mutant and its lowest touched layer."""
        c = self.cfg
        mutant = net.clone()
        widths = torch.tensor([float(w) for w in net.widths], device=net.device)
        layers = torch.multinomial(widths, c["mut"], replacement=True, generator=g)  # P(l) ~ width
        low = len(net.widths)
        for l in layers.tolist():
            w = net.widths[l]
            gate = torch.randint(w, (1,), generator=g, device=net.device)
            end = torch.randint(2, (1,), generator=g, device=net.device)
            src = torch.randint(net.offs[l], (1,), generator=g, device=net.device)
            mutant.srcs[l][end, gate] = src
            low = min(low, l)
        return mutant, low

    def train(self, data: Mnist, *, device: str = "cpu", seed: int = 0) -> None:
        c = self.cfg
        g = torch.Generator(device=device).manual_seed(seed)
        net = NandNet(c["bits"], c["widths"], device, g)

        # the harness speaks numpy; torch starts here
        enc_tr = self._encode(_t(data.train_x, device), net)  # (n_in, N) bool, encoded once
        y_tr = _t(data.train_y, device)
        enc_va = self._encode(_t(data.val_x, device), net)
        y_va = _t(data.val_y, device)

        best_val, best_srcs = -1.0, [s.clone() for s in net.srcs]
        stale = 0  # evaluations since the last improvement -- the convergence test
        t0 = time.time()
        for gen in range(c["gens"]):
            idx = torch.randint(enc_tr.shape[1], (c["batch"],), generator=g, device=device)
            xb, yb = enc_tr[:, idx], y_tr[idx]

            acts = net.forward(xb)  # the incumbent, computed once and reused by every mutant
            best_fit = margin(net.votes(acts), yb)
            winner = None
            for _ in range(c["k"] - 1):
                mutant, low = self._mutate(net, g)
                m_acts = mutant.forward(xb, acts.clone(), start=low)  # delta: layers < low reused
                fit = margin(mutant.votes(m_acts), yb)
                if fit > best_fit:  # strictly better, so the incumbent survives ties
                    best_fit, winner = fit, mutant
            if winner is not None:
                net = winner

            if (gen + 1) % c["eval_every"] == 0 or gen + 1 == c["gens"]:
                acc = self._accuracy(net, enc_va, y_va)
                if acc > best_val:
                    best_val, best_srcs, stale = acc, [s.clone() for s in net.srcs], 0
                else:
                    stale += 1
                print(f"  gen {gen + 1:6d}/{c['gens']}  margin {best_fit:+.3f}  "
                      f"val {acc:.2f}%  (best {best_val:.2f}%, stale {stale})  "
                      f"{(gen + 1) / (time.time() - t0):.0f} gen/s", flush=True)
                if stale >= c["patience"]:  # converged: no new best in patience*eval_every gens
                    print(f"  early stop at gen {gen + 1}: converged (best {best_val:.2f}%)",
                          flush=True)
                    break

        net.srcs = best_srcs
        self.net = net

    # ---- evaluation ----------------------------------------------------------------
    @staticmethod
    def _encode(pix: torch.Tensor, net: NandNet) -> torch.Tensor:
        """(N, 784) uint8 -> (n_in, N) bool, laid out exactly as hw.emit_thermometer."""
        t = torch.tensor(net.thresholds, device=pix.device, dtype=torch.int16)
        bits = pix.to(torch.int16).unsqueeze(-1) > t  # (N, 784, bits), pixel-major
        return bits.reshape(pix.shape[0], -1).T.contiguous()

    @torch.no_grad()
    def _accuracy(self, net: NandNet, enc: torch.Tensor, y: torch.Tensor, chunk: int = 4096):
        right = 0
        for i in range(0, enc.shape[1], chunk):
            v = net.votes(net.forward(enc[:, i : i + chunk]))
            right += (v.argmax(1) == y[i : i + chunk]).sum().item()
        return right / enc.shape[1] * 100

    @torch.no_grad()
    def predict(self, pix: np.ndarray) -> np.ndarray:
        net = self.net
        enc = self._encode(_t(pix, net.device), net)
        out = []
        for i in range(0, enc.shape[1], 4096):
            v = net.votes(net.forward(enc[:, i : i + 4096]))
            out.append(v.argmax(1).cpu())  # ties -> lowest class, same as the emitted argmax
        return torch.cat(out).numpy()

    @torch.no_grad()
    def scores(self, pix: np.ndarray) -> np.ndarray:
        """Per-class firing fraction in [0, 1]: votes() counts, divided by the gates per group."""
        net = self.net
        g = net.widths[-1] // N_CLASSES  # readout gates per class
        enc = self._encode(_t(pix, net.device), net)
        out = []
        for i in range(0, enc.shape[1], 4096):
            v = net.votes(net.forward(enc[:, i : i + 4096])) / g  # (B, 10) in [0, 1]
            out.append(v.cpu())
        return torch.cat(out).numpy()

    def emit_verilog(self) -> str:
        net = self.net
        layers = [
            (s[0].cpu(), s[1].cpu(), torch.full((s.shape[1],), NAND_TT)) for s in net.srcs
        ]
        return emit_lutnet(net.thresholds, layers)


def build(**point) -> Submission:
    return GeneticNand(**point)
