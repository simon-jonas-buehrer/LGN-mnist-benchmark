"""Backprop (STE) baseline for the bit-GA — the SAME fixed-wiring LUT net on the SAME MNIST pipeline.

This is the apples-to-apples counterpart to `ga_bits_mnist.py` (the fixed-wiring GA): identical
thermometer input, identical net shape (fixed random fan-in-2 wiring, learned 4-bit truth tables,
GroupSum popcount head), identical multilinear gate. The ONLY difference is how the truth tables are
trained — here by gradient descent with a straight-through `sin` surrogate (the DiffLogic/tutorial
parametrization), not by evolution. The LUT layer is lifted from the lut-tutorial `model.py`.

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


class LUTLayer(nn.Module):
    """Dense layer of 2-input LUT gates: fixed random fan-in-2 wiring, learned 4-bit truth tables."""

    def __init__(self, in_dim: int, out_dim: int, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        idx_a = torch.randint(in_dim, (out_dim,), generator=gen)
        idx_b = torch.randint(in_dim, (out_dim,), generator=gen)
        clash = idx_a == idx_b
        idx_b[clash] = (idx_b[clash] + 1) % in_dim
        self.register_buffer("idx_a", idx_a)
        self.register_buffer("idx_b", idx_b)
        self.weight = nn.Parameter(torch.randn(out_dim, 4, generator=gen))

    def forward(self, x):
        a, b = x[:, self.idx_a], x[:, self.idx_b]
        c = ste_bit(self.weight)
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


def build_net(n_in: int, widths: list[int], classes: int, seed: int) -> nn.Sequential:
    if widths[-1] % classes:
        raise ValueError(f"last width {widths[-1]} must be divisible by classes {classes}")
    layers, prev = [], n_in
    for i, w in enumerate(widths):
        layers.append(LUTLayer(prev, w, seed=seed + i))
        prev = w
    layers.append(GroupSum(classes))
    return nn.Sequential(*layers)


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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--run-name", type=str, default="backprop")
    p.add_argument("--metrics-out", type=str, default="")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    Xtr, ytr, Xte, yte = load_data_bits(a.dataset, a.device)
    n_in, classes = Xtr.shape[1], 10
    net = build_net(n_in, a.widths, classes, a.seed).to(a.device)
    n_gates = sum(a.widths)
    n_params = sum(q.numel() for q in net.parameters())  # 4 latents per gate
    print(
        f"net {n_in} -> {a.widths} ({n_gates} gates, {n_params} latents) device={a.device}",
        flush=True,
    )

    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    t0 = time.time()
    best = 0.0
    for ep in range(1, a.epochs + 1):
        net.train()
        perm = torch.randperm(len(Xtr), device=a.device)
        for i in range(0, len(Xtr), a.batch_size):
            idx = perm[i : i + a.batch_size]
            loss = F.cross_entropy(net(Xtr[idx]), ytr[idx])
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

    # Cost accounting in the GA's currency. Deployable = 4-bit truth tables only; the wiring is
    # fixed/random (structural, regenerable from the seed), so it is NOT stored -> ~2.3 KB, same as
    # the fixed-wiring GA. Backprop counts the backward pass: train_flops ~ 3x the forward gate-evals
    # (1 fwd + ~2 bwd), and there is no population factor.
    samples_seen = a.epochs * len(Xtr)  # images seen (steps * batch)
    fwd_gate_evals = samples_seen * n_gates
    metrics = {
        "run_name": a.run_name,
        "method": "backprop-ste",
        "dataset": a.dataset,
        "test_acc": round(test_acc, 4),
        "model_memory_bytes": math.ceil(n_gates * 4 / 8),  # tables only; wiring structural
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
