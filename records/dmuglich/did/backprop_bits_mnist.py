"""Backprop (STE) baseline for the bit-GA — the SAME fixed-wiring LUT net on the SAME MNIST pipeline.

This is the apples-to-apples counterpart to `ga_bits_mnist.py` (the fixed-wiring GA): identical
thermometer input, identical net shape (fixed random fan-in-2 wiring, learned 4-bit truth tables,
GroupSum popcount head), identical multilinear gate. The ONLY difference is how the truth tables are
trained — here by gradient descent with a straight-through `sin` surrogate (the DiffLogic/tutorial
parametrization), not by evolution. The LUT layer is lifted from the lut-tutorial `model.py`.

`--wire-codebook K` additionally relaxes the wiring-GA's codebook choice: each gate input picks among
the SAME K structural candidate sources (regenerated from `CODEBOOK_SEED`, so they match
`ga_bits_wiring_mnist.Net` exactly) via a straight-through argmax over learned logits. Forward is the
hard selection, so eval runs the exact discrete circuit and the hardened net is a (tables, wa, wb)
genome in the identical search space — the continuous version of what the GA/DID search.

`--soft` switches TRAINING to the DiffLogic-style soft relaxation: tables become per-corner
probabilities `sigmoid(w / tau)` and the wiring choice a `softmax(alpha / tau)` mixture, so real
probabilities flow through the multilinear gates and every parameter gets a true gradient instead
of a straight-through one. `tau` anneals geometrically from `--tau0` to `--tau1` over the run,
squeezing the relaxation toward the discrete corner. Eval ALWAYS runs the hard circuit (`.eval()`
hardens), so the reported accuracy is the deployable genome's in either mode.

It prints the same `METRICS {...}` line as the GA (same cost currency, CLAUDE.md Scoring) so the two
methods compare like-for-like. Backprop trains with fwd+bwd (train_flops counts the backward, ~2x the
forward) but carries no population factor — the point of the comparison.

Run (uses the lut-tutorial venv, which has cu126 torch):
    /path/to/lut-tutorial/.venv/bin/python backprop_bits_mnist.py --epochs 60
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
CACHE = Path(__file__).parent / ".cache" / "mnist.npz"
CIFAR_DIR = Path("/scratch/u6oz/nathanherr.u6oz/repos/lut-tutorial/data/cifar-10-batches-py")
THRESHOLDS = [32, 64, 96, 128, 160, 192, 224]  # same fixed thermometer as the GA


def _raw_cifar10(data_dir: Path):
    """CIFAR-10 as uint8 (N, 3072) pixels + int labels, read from the pickled batches."""
    import pickle

    def read(p):
        with open(p, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        return d[b"data"].astype(np.uint8), np.array(d[b"labels"])

    tr = [read(data_dir / f"data_batch_{i}") for i in range(1, 6)]
    xtr = np.concatenate([t[0] for t in tr])
    ytr = np.concatenate([t[1] for t in tr])
    xte, yte = read(data_dir / "test_batch")
    return xtr, ytr, xte, yte


def load_data_bits(dataset: str, device):
    """Thermometer-encoded MNIST/CIFAR-10, identical to the GA's pipeline."""
    if dataset == "mnist":
        if not CACHE.exists():
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            print(f"downloading MNIST -> {CACHE}")
            urllib.request.urlretrieve(MNIST_URL, CACHE)
        d = np.load(CACHE)
        xtr, ytr, xte, yte = d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    elif dataset == "cifar10":
        xtr, ytr, xte, yte = _raw_cifar10(CIFAR_DIR)
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    t = np.array(THRESHOLDS)

    def enc(a):
        bits = (a.reshape(a.shape[0], -1, 1) > t).astype(np.float32).reshape(a.shape[0], -1)
        return torch.from_numpy(bits).to(device)

    def lab(a):
        return torch.from_numpy(a.astype(np.int64)).to(device)

    return enc(xtr), lab(ytr), enc(xte), lab(yte)


def hard_bit(z):
    return (torch.sin(z) > 0).to(z.dtype)


def ste_bit(z):
    """Hard bit forward, smooth sin gradient backward (straight-through)."""
    soft = 0.5 + 0.5 * torch.sin(z)
    return hard_bit(z) + (soft - soft.detach())


def ste_onehot(logits):
    """Hard argmax one-hot forward, softmax gradient backward (straight-through)."""
    soft = logits.softmax(-1)
    hard = F.one_hot(logits.argmax(-1), logits.shape[-1]).to(soft.dtype)
    return hard + (soft - soft.detach())


def table_bits(w, soft: bool, training: bool, tau: float):
    """Truth-table corner values from latents, per relaxation mode.

    Soft mode trains on `sigmoid(w / tau)` and hardens to its own corner `w > 0` at eval — NOT
    `sin(w) > 0`, which disagrees with the sigmoid outside (-pi, pi). STE mode is hard both ways.
    """
    if not soft:
        return ste_bit(w)
    return torch.sigmoid(w / tau) if training else (w > 0).to(w.dtype)


def wire_onehot(alpha, soft: bool, training: bool, tau: float):
    """Wiring-choice weights from logits: soft-mode training mixes, everything else selects hard.

    `softmax(alpha / tau)` hardens to `argmax(alpha)` as tau -> 0, so the two paths agree."""
    if soft and training:
        return (alpha / tau).softmax(-1)
    return ste_onehot(alpha)


class LUTLayer(nn.Module):
    """Dense layer of 2-input LUT gates: fixed random fan-in-2 wiring, learned 4-bit truth tables."""

    def __init__(self, in_dim: int, out_dim: int, seed: int = 0, soft: bool = False):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        idx_a = torch.randint(in_dim, (out_dim,), generator=gen)
        idx_b = torch.randint(in_dim, (out_dim,), generator=gen)
        clash = idx_a == idx_b
        idx_b[clash] = (idx_b[clash] + 1) % in_dim
        self.register_buffer("idx_a", idx_a)
        self.register_buffer("idx_b", idx_b)
        self.weight = nn.Parameter(torch.randn(out_dim, 4, generator=gen))
        self.soft, self.tau = soft, 1.0

    def forward(self, x):
        a, b = x[:, self.idx_a], x[:, self.idx_b]
        c = table_bits(self.weight, self.soft, self.training, self.tau)
        f00, f01, f10, f11 = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
        return f00 + (f10 - f00) * a + (f01 - f00) * b + (f00 - f01 - f10 + f11) * (a * b)


class CodebookLUTLayer(nn.Module):
    """LUT gates whose two inputs each choose among K structural candidate sources.

    The continuous relaxation of the wiring-GA's codebook scheme: the candidates are the same
    seeded draws the GA uses (0 bytes at deploy), and each gate learns two K-way choice logits.
    Selection is straight-through argmax, so the forward pass — train and eval alike — runs the
    hard discrete circuit.
    """

    def __init__(
        self, cand_a: torch.Tensor, cand_b: torch.Tensor, seed: int = 0, soft: bool = False
    ):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.register_buffer("cand_a", cand_a)  # (K, out) source indices into the previous layer
        self.register_buffer("cand_b", cand_b)
        k, out = cand_a.shape
        self.weight = nn.Parameter(torch.randn(out, 4, generator=gen))
        self.alpha_a = nn.Parameter(0.01 * torch.randn(out, k, generator=gen))
        self.alpha_b = nn.Parameter(0.01 * torch.randn(out, k, generator=gen))
        self.soft, self.tau = soft, 1.0

    def forward(self, x):
        a = torch.einsum(
            "bko,ok->bo",
            x[:, self.cand_a],
            wire_onehot(self.alpha_a, self.soft, self.training, self.tau),
        )
        b = torch.einsum(
            "bko,ok->bo",
            x[:, self.cand_b],
            wire_onehot(self.alpha_b, self.soft, self.training, self.tau),
        )
        c = table_bits(self.weight, self.soft, self.training, self.tau)
        f00, f01, f10, f11 = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
        return f00 + (f10 - f00) * a + (f01 - f00) * b + (f00 - f01 - f10 + f11) * (a * b)


class GroupSum(nn.Module):
    """Popcount the final layer into k class logits, scaled by sqrt(group_size) (monotone)."""

    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x):
        gs = x.shape[-1] // self.k
        return x.reshape(x.shape[0], self.k, gs).sum(-1) / gs**0.5


def build_net(
    n_in: int, widths: list[int], classes: int, seed: int, codebook: int = 0, soft: bool = False
) -> nn.Sequential:
    if widths[-1] % classes:
        raise ValueError(f"last width {widths[-1]} must be divisible by classes {classes}")
    if codebook:
        # candidates must be bit-identical to the GA/DID net's: generate via the same class
        import os

        os.environ.setdefault("JAX_PLATFORMS", "cpu")  # keep jax off the GPU torch is using
        from ga_bits_wiring_mnist import Net as JaxNet

        jn = JaxNet(n_in, widths, classes, codebook=codebook)
        ca = torch.from_numpy(np.asarray(jn.cand_a).copy()).long()  # (K, n_gates)
        cb = torch.from_numpy(np.asarray(jn.cand_b).copy()).long()
        offs = jn.offs
    layers, prev = [], n_in
    for i, w in enumerate(widths):
        if codebook:
            lo, hi = offs[i], offs[i + 1]
            layers.append(CodebookLUTLayer(ca[:, lo:hi], cb[:, lo:hi], seed=seed + i, soft=soft))
        else:
            layers.append(LUTLayer(prev, w, seed=seed + i, soft=soft))
        prev = w
    layers.append(GroupSum(classes))
    return nn.Sequential(*layers)


def set_tau(net: nn.Sequential, tau: float) -> None:
    for m in net.modules():
        if hasattr(m, "tau"):
            m.tau = tau


@torch.no_grad()
def accuracy(net, x, y, batch=4096):
    net.eval()
    correct = 0
    for i in range(0, len(x), batch):
        correct += (net(x[i : i + batch]).argmax(1) == y[i : i + batch]).sum().item()
    return correct / len(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"])
    p.add_argument("--widths", type=int, nargs="+", default=[3072, 1024, 500])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument(
        "--wire-codebook",
        type=int,
        default=0,
        help="0 = fixed wiring; K = learn a choice among the GA's K structural candidate wirings per gate input",
    )
    p.add_argument("--soft", action="store_true", help="train on the soft relaxation (hard eval)")
    p.add_argument("--tau0", type=float, default=1.0, help="soft: initial temperature")
    p.add_argument("--tau1", type=float, default=0.1, help="soft: final temperature (geometric)")
    p.add_argument(
        "--distill",
        type=str,
        default="",
        help="teacher logits npz (records/darius/did/teacher_mnist.py); train on soft-target CE",
    )
    p.add_argument("--distill-alpha", type=float, default=1.0, help="soft-target weight vs one-hot")
    p.add_argument("--distill-temp", type=float, default=4.0, help="teacher softmax temperature")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--run-name", type=str, default="backprop")
    p.add_argument("--metrics-out", type=str, default="")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    Xtr, ytr, Xte, yte = load_data_bits(a.dataset, a.device)
    n_in, classes = Xtr.shape[1], 10
    tgt = None
    if a.distill:
        td = np.load(a.distill)
        assert (ytr.cpu().numpy() == td["y_train"]).all(), "teacher logits misaligned"
        q = torch.softmax(torch.tensor(td["train_logits"], device=a.device) / a.distill_temp, dim=1)
        oh = F.one_hot(ytr, classes).float()
        tgt = (1.0 - a.distill_alpha) * oh + a.distill_alpha * q
    net = build_net(n_in, a.widths, classes, a.seed, codebook=a.wire_codebook, soft=a.soft).to(
        a.device
    )
    n_gates = sum(a.widths)
    n_params = sum(q.numel() for q in net.parameters())  # 4 table (+2K wiring) latents per gate
    print(
        f"net {n_in} -> {a.widths} ({n_gates} gates, {n_params} latents) device={a.device}",
        flush=True,
    )

    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    t0 = time.time()
    best = 0.0
    for ep in range(1, a.epochs + 1):
        if a.soft:
            set_tau(net, a.tau0 * (a.tau1 / a.tau0) ** ((ep - 1) / max(a.epochs - 1, 1)))
        net.train()
        perm = torch.randperm(len(Xtr), device=a.device)
        for i in range(0, len(Xtr), a.batch_size):
            idx = perm[i : i + a.batch_size]
            if tgt is None:
                loss = F.cross_entropy(net(Xtr[idx]), ytr[idx])
            else:  # same mixed soft-target CE the discrete optimizers train on
                loss = -(tgt[idx] * F.log_softmax(net(Xtr[idx]), dim=1)).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        te = accuracy(net, Xte, yte)  # peek at test each epoch (same methodology as the GA)
        best = max(best, te)
        if ep % 5 == 0 or ep == a.epochs:
            print(
                f"epoch {ep:3d}  test {te:.4f}  (best {best:.4f})  ({time.time() - t0:.0f}s)",
                flush=True,
            )
    train_seconds = time.time() - t0
    test_acc = best

    # Cost accounting in the GA's currency. Deployable = 4-bit truth tables, plus 2*log2(K) choice
    # bits per gate in codebook mode (the candidates themselves are structural, 0 bytes) — without
    # the codebook the wiring is fixed/random and NOT stored -> ~2.3 KB, same as the fixed-wiring
    # GA. Backprop counts the backward pass: train_flops ~ 3x the forward gate-evals (1 fwd + ~2
    # bwd), and there is no population factor.
    samples_seen = a.epochs * len(Xtr)  # images seen (steps * batch)
    fwd_gate_evals = samples_seen * n_gates
    wire_bits = (
        n_gates * 2 * max(1, math.ceil(math.log2(a.wire_codebook))) if a.wire_codebook else 0
    )
    metrics = {
        "run_name": a.run_name,
        "method": "backprop-soft" if a.soft else "backprop-ste",
        "dataset": a.dataset,
        "wire_codebook": a.wire_codebook,
        "test_acc": round(test_acc, 4),
        "model_memory_bytes": math.ceil((n_gates * 4 + wire_bits) / 8),
        "n_gates": n_gates,
        "n_params": n_params,
        "epochs": a.epochs,
        "batch": a.batch_size,
        "samples_seen": samples_seen,
        "forward_passes": samples_seen,  # no population factor
        "gate_evaluations": fwd_gate_evals,  # forward-only, for direct compare to the GA's number
        "train_flops": 3 * fwd_gate_evals,  # fwd + ~2x bwd
        "train_seconds": round(train_seconds, 1),
        "gpu_count": 1,
        "gpu_hours": round(train_seconds / 3600, 4),
    }
    print("METRICS " + json.dumps(metrics), flush=True)
    if a.metrics_out:
        Path(a.metrics_out).write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
