"""Reload the trained checkpoint and evaluate it - shows how to use the saved weights.

    uv run python load.py                 # loads lut_cifar10.pt, prints test accuracy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import Config, LUTNet, Thermometer
from train import evaluate, load_cifar10


def load_model(ckpt_path: Path, device: str = "cpu") -> LUTNet:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = Config(**ckpt["config"])
    # Build with a dummy fitted thermometer; the real thresholds come from state_dict.
    enc = Thermometer(num_bits=cfg.num_bits)
    enc.thresholds = torch.zeros(cfg.in_channels, cfg.num_bits)
    model = LUTNet(cfg, encoder=enc)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=Path("results/lut_cifar10.pt"))
    p.add_argument("--data-dir", type=Path, default=Path("data/cifar-10-batches-py"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    model = load_model(args.ckpt, args.device)
    _, _, test_x, test_y = load_cifar10(args.data_dir, download=False)
    m = evaluate(model, test_x, test_y, args.device)
    print(f"test: loss={m['loss']:.3f}  acc={m['acc']:.2f}%  ppl={m['ppl']:.2f}")

    # Peek at what one neuron learned: its 4-entry boolean truth table.
    first_lut = next(m for m in model if hasattr(m, "truth_table"))
    print("truth table of neuron 0 (f00,f01,f10,f11):",
          first_lut.truth_table()[0].int().tolist())


if __name__ == "__main__":
    main()
