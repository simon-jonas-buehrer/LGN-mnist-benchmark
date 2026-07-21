"""Train a small CNN teacher on MNIST and export its logits for distillation.

The rewire-DID harness consumes the saved npz via --distill: soft targets
softmax(train_logits / T) replace the one-hot label rows in the head loss and the lambda seed.
Logits are exported in dataset order on the untouched images (no augmentation at export), with
y_train stored as an alignment fingerprint the harness asserts against.

Run (GPU, Slurm):
    python teacher_mnist.py --out .cache/teacher_mnist.npz
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tyro
from torch import nn

CACHE = Path(__file__).resolve().parents[2] / "nherr" / "mnist-ga" / ".cache" / "mnist.npz"
MEAN, STD = 0.1307, 0.3081


@dataclass
class Config:
    epochs: int = 12
    batch: int = 256
    lr: float = 1e-3
    shift: int = 2  # translation augmentation in pixels, one draw per batch (0 = off)
    out: str = ".cache/teacher_mnist.npz"
    seed: int = 0


class Cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


def logits_in_order(model, x, device, batch=1024):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out.append(model(x[i : i + batch].to(device)).float().cpu())
    return torch.cat(out).numpy().astype(np.float32)


def main(cfg: Config) -> None:
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(CACHE)
    xtr = torch.tensor((d["x_train"] / 255.0 - MEAN) / STD, dtype=torch.float32)[:, None]
    ytr = torch.tensor(d["y_train"], dtype=torch.long)
    xte = torch.tensor((d["x_test"] / 255.0 - MEAN) / STD, dtype=torch.float32)[:, None]

    model = Cnn().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    n = len(xtr)
    for ep in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n)
        tot = correct = 0
        for i in range(0, n, cfg.batch):
            xb = xtr[perm[i : i + cfg.batch]].to(device)
            yb = ytr[perm[i : i + cfg.batch]].to(device)
            if cfg.shift:
                s = cfg.shift
                dy, dx = torch.randint(-s, s + 1, (2,)).tolist()
                xb = F.pad(xb, (s, s, s, s))[:, :, s - dy : s - dy + 28, s - dx : s - dx + 28]
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
        sched.step()
        print(f"epoch {ep + 1:2d}/{cfg.epochs}  train acc {correct / tot:.4f}", flush=True)

    train_logits = logits_in_order(model, xtr, device)
    test_logits = logits_in_order(model, xte, device)
    train_acc = float((train_logits.argmax(1) == d["y_train"]).mean())
    test_acc = float((test_logits.argmax(1) == d["y_test"]).mean())
    print(f"teacher: train {train_acc:.4f}  TEST {test_acc:.4f}", flush=True)
    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        train_logits=train_logits,
        test_logits=test_logits,
        y_train=d["y_train"],
        test_acc=np.float32(test_acc),
    )
    print(f"saved logits -> {out}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Config))
