"""A small LUT network in plain PyTorch.

A LUT (look-up table) network is a neural network whose neurons are boolean logic gates
rather than weighted sums. Everything between layers is a single bit. Each neuron reads a
fixed number of input bits (here two, so fan_in = 2) and applies a 2-input boolean function
to them, stored as a 4-entry truth table.

A 2-input LUT has 2**2 = 4 entries (the outputs for inputs 00, 01, 10, 11). Four free bits
per neuron cover all 2**4 = 16 boolean functions of two inputs (AND, OR, XOR, NAND, pass-A,
constant-1, and so on), so a neuron learns which gate it is. Stacking layers gives a deep
combinational circuit that trains by gradient descent and maps onto FPGA/ASIC LUTs at
inference.

Pipeline, all binary:

    image (uint8 pixels)
      -> Thermometer encoder        real pixel -> a few threshold bits
      -> Flatten                    (B, n_bits)
      -> several LUTLayers          each neuron is a learned 2-input gate, fan-in 2
      -> GroupSum head              popcount the last layer into class scores
      -> logits (B, 10)

Backprop through a boolean gate uses the light parametrization with a sin activation: the
forward pass is a hard bit, the backward pass sees a smooth sin surrogate (see LUTLayer).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


# ======================================================================================
# 1. sin activation: a real-valued latent maps to a learnable bit
# ======================================================================================
#
# Each learnable bit is stored as a real number z (a latent), turned into a bit by the sign
# of sin(z):
#
#       hard bit  =  1[sin(z) > 0]              exact 0 or 1, used in the forward pass
#       soft bit  =  0.5 + 0.5 * sin(z)         smooth in [0, 1], used for the gradient
#
# sin is periodic, so the latent never saturates: there is always a gradient toward the
# nearest 0 or 1 basin. This is a drop-in replacement for sigmoid in the light LUT
# parametrization.


def hard_bit(z: torch.Tensor) -> torch.Tensor:
    """Exact {0, 1} bit from a latent: 1 where sin(z) > 0, else 0."""
    return (torch.sin(z) > 0).to(z.dtype)


def soft_bit(z: torch.Tensor) -> torch.Tensor:
    """Smooth surrogate of hard_bit, valued in [0, 1]."""
    return 0.5 + 0.5 * torch.sin(z)


def ste_bit(z: torch.Tensor) -> torch.Tensor:
    """Straight-through binarizer: hard bit forward, smooth sin gradient backward.

    soft - soft.detach() is zero in the forward pass, so the returned value is exactly the
    hard bit, while its gradient is the gradient of soft_bit.
    """
    soft = soft_bit(z)
    return hard_bit(z) + (soft - soft.detach())


# ======================================================================================
# 2. LUT layer: a layer of learned 2-input boolean gates
# ======================================================================================


class LUTLayer(nn.Module):
    """A dense layer of out_dim neurons, each a learned 2-input LUT (fan-in 2).

    Two pieces per layer:

      Connections (fixed, not learned). For each neuron we pick which 2 of the in_dim input
      bits it reads. They are sampled once at random and frozen, the fixed random wiring used
      by DiffLogic. They live in a buffer, so they are saved with the model and reproducible.

      The LUT (learned). weight has shape (out_dim, 4): four latents per neuron, one per
      truth-table entry f(0,0), f(0,1), f(1,0), f(1,1). The sin activation turns them into
      the 4 bits of the gate's truth table.

    The forward uses the multilinear form of a 2-input LUT. For bits a, b and truth-table
    values f00, f01, f10, f11:

        out = f00 + (f10 - f00)*a + (f01 - f00)*b + (f00 - f01 - f10 + f11)*a*b

    For exact {0,1} bits this returns the selected truth-table entry, a real boolean gate;
    for the smooth surrogates it interpolates so gradients flow. The straight-through bit
    makes train and eval compute the same hard bits, eval just skips the surrogate.
    """

    def __init__(self, in_dim: int, out_dim: int, *, seed: int = 0) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Fixed random fan-in-2 wiring: which 2 inputs each neuron reads.
        gen = torch.Generator().manual_seed(seed)
        idx_a = torch.randint(in_dim, (out_dim,), generator=gen)
        idx_b = torch.randint(in_dim, (out_dim,), generator=gen)
        # Avoid a neuron reading the same input twice (a gate of (x, x) is wasteful).
        clash = idx_a == idx_b
        idx_b[clash] = (idx_b[clash] + 1) % in_dim
        self.register_buffer("idx_a", idx_a)
        self.register_buffer("idx_b", idx_b)

        # Learnable truth tables: 4 latents per neuron, random init.
        self.weight = nn.Parameter(torch.randn(out_dim, 4, generator=gen))

    def truth_table(self) -> torch.Tensor:
        """Discrete {0,1} truth tables (out_dim, 4), for inspection or export."""
        return hard_bit(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim) of bits. Gather the 2 inputs each neuron reads -> (B, out_dim).
        a = x[:, self.idx_a]
        b = x[:, self.idx_b]

        # Truth-table values per neuron, as bits (hard forward, soft gradient).
        c = ste_bit(self.weight)  # (out_dim, 4) = [f00, f01, f10, f11]
        f00, f01, f10, f11 = c[:, 0], c[:, 1], c[:, 2], c[:, 3]

        # Multilinear LUT evaluation, broadcast over the batch.
        return f00 + (f10 - f00) * a + (f01 - f00) * b + (f00 - f01 - f10 + f11) * (a * b)

    def extra_repr(self) -> str:
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}, fan_in=2"


# ======================================================================================
# 3. Thermometer encoder: real pixels to bits
# ======================================================================================


class Thermometer(nn.Module):
    """Encode each input feature as num_bits threshold bits (a thermometer code).

    For a value v and thresholds t_1 < t_2 < ... < t_k the code is
    [v > t_1, v > t_2, ..., v > t_k], a monotone, unary-like representation. Thresholds are
    the per-channel quantiles of the fit data (a distributive thermometer), so each bit
    splits roughly the same number of pixels.

    Call fit() once on a sample of training images before use. Input is (B, C, H, W) and
    output is (B, C*num_bits, H, W) bits.
    """

    def __init__(self, num_bits: int = 2) -> None:
        super().__init__()
        self.num_bits = num_bits
        self.register_buffer("thresholds", None, persistent=True)  # (C, num_bits)

    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> "Thermometer":
        # x: (N, C, H, W). Per-channel quantile thresholds.
        c = x.shape[1]
        flat = x.permute(1, 0, 2, 3).reshape(c, -1)  # (C, N*H*W)
        qs = torch.linspace(0, 1, self.num_bits + 2)[1:-1]  # drop 0 and 1
        self.thresholds = torch.quantile(flat, qs.to(flat), dim=1).T.contiguous()  # (C, num_bits)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.thresholds is None:
            raise RuntimeError("Thermometer must be .fit() on data before use")
        # x: (B, C, H, W) compared against (1, C, 1, 1, num_bits)
        t = self.thresholds.view(1, x.shape[1], 1, 1, self.num_bits)
        bits = (x.unsqueeze(-1) > t).to(x.dtype)  # (B, C, H, W, num_bits)
        # Fold num_bits into the channel axis: (B, C*num_bits, H, W)
        b, c, h, w, k = bits.shape
        return bits.permute(0, 1, 4, 2, 3).reshape(b, c * k, h, w)


# ======================================================================================
# 4. GroupSum head: read bits out as class scores
# ======================================================================================


class GroupSum(nn.Module):
    """Split the final bit vector into k equal groups and sum each group (popcount).

    The last LUT layer has out_dim bits; we cut them into k classes of
    group_size = out_dim // k bits and count the ones per class.

    Each group sum is a sum of group_size near-Bernoulli bits, so its standard deviation
    grows like sqrt(group_size). We divide by sqrt(group_size) (the variance-based scaling,
    the same idea as 1/sqrt(d) attention scaling) to keep the logit variance roughly constant
    no matter how wide the last layer is, so the softmax that follows is neither saturated nor
    washed out. The scaling is monotone, so it does not affect the argmax at eval. Pass an
    explicit tau to override the auto value.
    """

    def __init__(self, k: int, tau: float | None = None) -> None:
        super().__init__()
        self.k = k
        self.tau = tau  # None -> variance-based sqrt(group_size), computed per forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % self.k != 0:
            raise ValueError(f"last dim {x.shape[-1]} not divisible by k={self.k}")
        group_size = x.shape[-1] // self.k
        tau = self.tau if self.tau is not None else group_size**0.5
        groups = x.reshape(x.shape[0], self.k, group_size)
        return groups.sum(-1) / tau

    def extra_repr(self) -> str:
        return f"k={self.k}, tau={self.tau if self.tau is not None else 'sqrt(group_size)'}"


# ======================================================================================
# 5. The full model
# ======================================================================================


@dataclass
class Config:
    num_classes: int = 10
    in_channels: int = 3
    image_size: int = 32
    num_bits: int = 2          # thermometer bits per channel
    layer_widths: tuple[int, ...] = (12000, 12000, 12000)  # one LUTLayer per entry
    tau: float | None = None   # head logit scale; None = variance-based sqrt(group_size)
    seed: int = 0


class LUTNet(nn.Sequential):
    """Thermometer -> Flatten -> N x LUTLayer -> GroupSum.

    Build it with a fitted thermometer so the input bit-width is known:

        enc = Thermometer(num_bits=2).fit(train_images)
        model = LUTNet(Config(), encoder=enc)
    """

    def __init__(self, cfg: Config, encoder: Thermometer) -> None:
        self.cfg = cfg
        in_dim = cfg.in_channels * cfg.num_bits * cfg.image_size * cfg.image_size

        layers: list[nn.Module] = [encoder, nn.Flatten()]
        prev = in_dim
        for i, width in enumerate(cfg.layer_widths):
            if i == len(cfg.layer_widths) - 1 and width % cfg.num_classes != 0:
                raise ValueError(
                    f"last layer width {width} must be divisible by num_classes {cfg.num_classes}"
                )
            layers.append(LUTLayer(prev, width, seed=cfg.seed + i))
            prev = width
        layers.append(GroupSum(k=cfg.num_classes, tau=cfg.tau))
        super().__init__(*layers)


def build_model(cfg: Config, fit_samples: torch.Tensor) -> LUTNet:
    """Fit a thermometer on fit_samples and assemble the model."""
    enc = Thermometer(num_bits=cfg.num_bits).fit(fit_samples)
    return LUTNet(cfg, encoder=enc)
