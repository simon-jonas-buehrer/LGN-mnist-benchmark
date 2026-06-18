"""Train the tiny LUT network on CIFAR-10 and watch it OVERFIT.

The whole point of this script is pedagogical: there is NO regularization and NO data
augmentation. We train on a small subset of CIFAR-10 so that the model memorizes the
training set (train accuracy shoots toward 100%) while validation/test accuracy plateaus
far below. The widening gap between the curves IS overfitting.

Everything is written to a results folder (default ``results/``):
    results/train.log      full console log
    results/metrics.csv    per-epoch train/val/test loss, accuracy, perplexity
    results/curves.png     plots of loss, accuracy and perplexity over epochs
    results/lut_cifar10.pt trained weights (the checkpoint)

    uv run python train.py                       # overfit a 5k subset (default)
    uv run python train.py --train-size 0        # use the full training set instead
    uv run python train.py --epochs 50 --device cuda

CIFAR-10 is read directly from the original ``cifar-10-batches-py`` pickle files (no
torchvision dependency). Point ``--data-dir`` at a folder that contains ``data_batch_*``
and ``test_batch``, or pass ``--download`` to fetch+extract it there.
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F

from model import Config, build_model

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
VAL_SIZE = 5000  # images held out of training for validation (always disjoint from train)


# --------------------------------------------------------------------------------------
# Data: a minimal, dependency-free CIFAR-10 loader
# --------------------------------------------------------------------------------------
def _maybe_download(data_dir: Path) -> None:
    if (data_dir / "test_batch").exists():
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    tgz = data_dir / "cifar-10-python.tar.gz"
    print(f"downloading CIFAR-10 to {tgz} ...", flush=True)
    urllib.request.urlretrieve(CIFAR_URL, tgz)
    with tarfile.open(tgz) as tar:
        tar.extractall(data_dir)
    nested = data_dir / "cifar-10-batches-py"  # archive extracts to a nested dir - flatten it
    if nested.exists():
        for f in nested.iterdir():
            f.rename(data_dir / f.name)


def load_cifar10(data_dir: Path, download: bool) -> tuple[torch.Tensor, ...]:
    """Return ``(train_x, train_y, test_x, test_y)`` with images in ``[0,1]``, shape (N,3,32,32)."""
    if download:
        _maybe_download(data_dir)

    def read(batch: Path) -> tuple[torch.Tensor, torch.Tensor]:
        with open(batch, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        x = torch.tensor(d[b"data"], dtype=torch.float32).reshape(-1, 3, 32, 32) / 255.0
        y = torch.tensor(d[b"labels"], dtype=torch.long)
        return x, y

    train = [read(data_dir / f"data_batch_{i}") for i in range(1, 6)]
    train_x = torch.cat([t[0] for t in train])
    train_y = torch.cat([t[1] for t in train])
    test_x, test_y = read(data_dir / "test_batch")
    return train_x, train_y, test_x, test_y


# --------------------------------------------------------------------------------------
# Evaluation: loss, accuracy and perplexity over a dataset
# --------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, x, y, device: str, batch: int = 1000) -> dict[str, float]:
    model.eval()
    correct, loss_sum = 0, 0.0
    for i in range(0, len(x), batch):
        xb, yb = x[i : i + batch].to(device), y[i : i + batch].to(device)
        logits = model(xb)
        loss_sum += F.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(1) == yb).sum().item()
    loss = loss_sum / len(x)
    return {"loss": loss, "acc": 100.0 * correct / len(x), "ppl": math.exp(loss)}


# --------------------------------------------------------------------------------------
# Plot the learning curves
# --------------------------------------------------------------------------------------
def plot_curves(history: list[dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    splits = {"train": "tab:blue", "val": "tab:orange", "test": "tab:green"}
    metrics = [("loss", "cross-entropy loss"), ("acc", "accuracy (%)"), ("ppl", "perplexity")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (key, title) in zip(axes, metrics):
        for split, color in splits.items():
            ax.plot(epochs, [h[f"{split}_{key}"] for h in history], label=split, color=color)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("LUT network on CIFAR-10 - train/val/test (overfitting demo)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/cifar-10-batches-py"))
    p.add_argument("--download", action="store_true", help="download CIFAR-10 if missing")
    p.add_argument("--train-size", type=int, default=5000,
                   help="number of training images (0 = full set minus the val split)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--num-bits", type=int, default=2)
    p.add_argument("--width", type=int, default=12000, help="neurons per LUT layer")
    p.add_argument("--layers", type=int, default=3, help="number of LUT layers")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    args = p.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(args.results_dir / "train.log", "w")

    def log(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    torch.manual_seed(args.seed)
    device = args.device
    log(f"device={device}  args={vars(args)}")

    train_x, train_y, test_x, test_y = load_cifar10(args.data_dir, args.download)

    # Hold out a fixed validation split (disjoint from training).
    val_x, val_y = train_x[-VAL_SIZE:], train_y[-VAL_SIZE:]
    pool_x, pool_y = train_x[:-VAL_SIZE], train_y[:-VAL_SIZE]
    if args.train_size > 0:
        pool_x, pool_y = pool_x[: args.train_size], pool_y[: args.train_size]
    log(f"train={len(pool_x)}  val={len(val_x)}  test={len(test_x)} images")

    cfg = Config(num_bits=args.num_bits, layer_widths=tuple([args.width] * args.layers),
                 seed=args.seed)
    model = build_model(cfg, pool_x[:2000]).to(device)  # fit thermometer on a sample
    log(model)
    log(f"learnable LUT latents: {sum(q.numel() for q in model.parameters()):,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)  # no weight decay -> no regularization

    header = ("epoch | tr_loss | tr_acc | tr_ppl | va_loss | va_acc | va_ppl | "
              "te_loss | te_acc | te_ppl | gap | time")
    log("\n" + header)
    log("-" * len(header))
    history: list[dict] = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(pool_x))
        for i in range(0, len(pool_x), args.batch_size):
            idx = perm[i : i + args.batch_size]
            xb, yb = pool_x[idx].to(device), pool_y[idx].to(device)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        tr = evaluate(model, pool_x, pool_y, device)
        va = evaluate(model, val_x, val_y, device)
        te = evaluate(model, test_x, test_y, device)
        row = {"epoch": epoch,
               **{f"train_{k}": v for k, v in tr.items()},
               **{f"val_{k}": v for k, v in va.items()},
               **{f"test_{k}": v for k, v in te.items()}}
        history.append(row)
        log(f"{epoch:5d} | {tr['loss']:7.3f} | {tr['acc']:6.2f} | {tr['ppl']:6.2f} | "
            f"{va['loss']:7.3f} | {va['acc']:6.2f} | {va['ppl']:6.2f} | "
            f"{te['loss']:7.3f} | {te['acc']:6.2f} | {te['ppl']:6.2f} | "
            f"{tr['acc'] - te['acc']:4.1f} | {time.time() - t0:4.0f}s")

    # --- write the metrics CSV, the plot, and the checkpoint ---
    with open(args.results_dir / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    plot_curves(history, args.results_dir / "curves.png")
    ckpt = args.results_dir / "lut_cifar10.pt"
    torch.save({"state_dict": model.state_dict(), "config": vars(cfg),
                "train_size": len(pool_x), "history": history}, ckpt)

    log(f"\nwrote: {args.results_dir}/train.log, metrics.csv, curves.png, {ckpt.name}")
    log("Notice the gap: train_acc climbs toward 100% while val/test_acc stall "
        "-> the LUT net is overfitting, exactly as intended.")
    log_file.close()


if __name__ == "__main__":
    sys.exit(main())
