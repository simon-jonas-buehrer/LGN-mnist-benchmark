"""MNIST, exactly as every submission sees it: plain numpy uint8.

Images stay uint8 (0..255) and flat, (N, 784), because that is what the circuit gets -- the top
module's input is the raw pixel bytes. Any float, threshold or normalization you want is part
of YOUR model, and lands in YOUR gate count.

numpy, not torch, so a submission can be written in torch, JAX, TensorFlow or nothing at all.

The split is fixed for everyone: the official 60k training images are cut 54k/6k into
train/val with a fixed permutation, and the official 10k test images are the test set. Train on
train, tune on val. You never need test -- the harness computes the leaderboard number itself,
by simulating your synthesized netlist.
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
         "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "mnist"

N_PIXELS = 784   # 28 x 28, row-major
PIXEL_BITS = 8   # one grayscale byte per pixel
N_CLASSES = 10
VAL_SIZE = 6000


@dataclass
class Mnist:
    """uint8 images (N, 784) and int64 labels (N,)."""

    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray


def _read(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    magic, n = struct.unpack(">II", raw[:8])
    if magic == 0x803:  # images
        h, w = struct.unpack(">II", raw[8:16])
        # .copy(): frombuffer is read-only, which torch.from_numpy complains about
        return np.frombuffer(raw, np.uint8, offset=16).reshape(n, h * w).copy()
    if magic == 0x801:  # labels
        return np.frombuffer(raw, np.uint8, offset=8).astype(np.int64)
    raise ValueError(f"{path}: bad idx magic {magic:#x}")


def load(data_dir: Path = DATA_DIR) -> Mnist:
    data_dir.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        if not (data_dir / f).exists():
            print(f"downloading {f} ...", flush=True)
            urllib.request.urlretrieve(MIRROR + f, data_dir / f)

    x, y = _read(data_dir / FILES[0]), _read(data_dir / FILES[1])
    perm = np.random.default_rng(0).permutation(len(x))
    val, train = perm[:VAL_SIZE], perm[VAL_SIZE:]
    return Mnist(x[train], y[train], x[val], y[val],
                 _read(data_dir / FILES[2]), _read(data_dir / FILES[3]))


def to_bits(pix: np.ndarray) -> np.ndarray:
    """(N, 784) uint8 pixels -> (N, 6272) uint8 bits, in the top module's port order.

    Bit 8*p + k is bit k (LSB first) of pixel p, i.e. exactly `pix[8*p +: 8]` in the Verilog.
    """
    bits = (pix[:, :, None] >> np.arange(PIXEL_BITS, dtype=np.uint8)) & 1
    return bits.reshape(len(pix), N_PIXELS * PIXEL_BITS)
