"""A tiny LUT network - pure PyTorch, no custom CUDA, no autograd tricks beyond a one-line STE.

A LUT (look-up table) network is a neural network whose neurons are *boolean logic
gates* instead of weighted sums. Everything that flows through the network is a bit
(a 0 or a 1). Each neuron:

  1. reads a small, FIXED number of input bits (here ``fan_in = 2``), and
  2. applies a 2-input boolean function - i.e. a 4-entry look-up table - to them.

A 2-input LUT has 2**2 = 4 entries (the truth table for inputs 00, 01, 10, 11). With
4 free bits per neuron you can represent ALL 2**4 = 16 boolean functions of two inputs
(AND, OR, XOR, NAND, "pass A", "constant 1", ...). So one neuron *learns which gate it
should be*. Stack layers of these and you get a deep combinational logic circuit that is
trained by gradient descent - and that maps directly onto FPGA/ASIC LUTs at inference.

The whole pipeline, all binary:

    image (uint8 pixels)
      -> Thermometer encoder        # real pixel -> a few threshold bits
      -> Flatten                    # (B, n_bits)
      -> several LUTLayers          # each neuron = a learned 2-input gate, fan-in 2
      -> GroupSum head              # popcount the last layer into class scores
      -> logits (B, 10)

The only non-obvious part is "how do you backprop through a boolean gate?". See
``LUTLayer`` below: we use the *light parametrization with a sin activation* - the
forward pass is an exact hard bit, the backward pass sees a smooth ``sin`` surrogate.

This file is deliberately self-contained and small. Read it top to bottom.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


# ======================================================================================
# 1. The "sin" activation - how a real-valued latent becomes a learnable bit
# ======================================================================================
#
# Every learnable bit in the network is stored as a real number ``z`` (a "latent").
# We turn it into a bit with the sign of sin(z):
#
#       hard bit  =  1[sin(z) > 0]              # exact 0 or 1  -> used in the forward pass
#       soft bit  =  0.5 + 0.5 * sin(z)         # smooth in [0, 1] -> used for the gradient
#
# Why sin and not the usual sigmoid? sin is periodic, so the latent never saturates: there
# is always a non-zero gradient pushing it toward the nearest "0" or "1" basin. This is the
# parametrization used in the binary-attention record (lut-golf). It is a drop-in swap for
# sigmoid in the standard "light" LUT parametrization.


def hard_bit(z: torch.Tensor) -> torch.Tensor:
    """Exact {0, 1} bit from a latent: ``1`` where ``sin(z) > 0`` else ``0``."""
    return (torch.sin(z) > 0).to(z.dtype)


def soft_bit(z: torch.Tensor) -> torch.Tensor:
    """Smooth surrogate of :func:`hard_bit`, valued in ``[0, 1]``."""
    return 0.5 + 0.5 * torch.sin(z)


def ste_bit(z: torch.Tensor) -> torch.Tensor:
    """Straight-through binarizer: EXACT hard bit forward, smooth ``sin`` gradient backward.

    ``soft - soft.detach()`` is numerically ``0.0`` in the forward pass (so the value
    returned is exactly the hard bit, no float dust), but its gradient is the gradient of
    ``soft_bit``. Standard straight-through estimator (STE) trick.
    """
    soft = soft_bit(z)
    return hard_bit(z) + (soft - soft.detach())


# ======================================================================================
# 2. The LUT layer - a layer of learned 2-input boolean gates
# ======================================================================================


class LUTLayer(nn.Module):
    """A dense layer of ``out_dim`` neurons, each a learned 2-input LUT (fan-in 2).

    Two pieces per layer:

      * **connections** (fixed, not learned) - for each neuron we pick which 2 of the
        ``in_dim`` input bits it reads. We pick them once at random and freeze them
        (this is the "fixed random wiring" used by DiffLogic). They live in a buffer, so
        they are saved with the model and reproducible.

      * **the LUT itself** (learned) - ``weight`` has shape ``(out_dim, 4)``: four latents
        per neuron, one per truth-table entry ``f(0,0), f(0,1), f(1,0), f(1,1)``. Passed
        through the sin activation they become the 4 bits of the gate's truth table.

    The forward evaluates the gate with the *multilinear* form of a 2-input LUT. For bits
    ``a, b`` and truth-table values ``f00, f01, f10, f11``:

        out = f00 + (f10 - f00)*a + (f01 - f00)*b + (f00 - f01 - f10 + f11)*a*b

    When ``a, b`` and the ``f``'s are exact {0,1} bits this returns exactly the truth-table
    entry selected by ``(a, b)`` - i.e. a real boolean gate. When they are the smooth
    surrogates it interpolates, which is what lets gradients flow. Because we use the STE
    everywhere, ``train`` and ``eval`` compute the *same* hard bits; ``eval`` just skips the
    surrogate.
    """

    def __init__(self, in_dim: int, out_dim: int, *, seed: int = 0) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # --- fixed random fan-in-2 wiring: which 2 inputs each neuron reads ---
        gen = torch.Generator().manual_seed(seed)
        idx_a = torch.randint(in_dim, (out_dim,), generator=gen)
        idx_b = torch.randint(in_dim, (out_dim,), generator=gen)
        # Avoid a neuron reading the same input twice (a 2-input gate of (x, x) is wasteful).
        clash = idx_a == idx_b
        idx_b[clash] = (idx_b[clash] + 1) % in_dim
        self.register_buffer("idx_a", idx_a)
        self.register_buffer("idx_b", idx_b)

        # --- the learnable truth tables: 4 latents per neuron ---
        # Random init so each gate starts as a random-ish boolean function.
        self.weight = nn.Parameter(torch.randn(out_dim, 4, generator=gen))

    def truth_table(self) -> torch.Tensor:
        """Return the discrete ``{0,1}`` truth tables ``(out_dim, 4)`` - for inspection/export."""
        return hard_bit(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim) of bits. Gather the 2 inputs each neuron reads -> (B, out_dim).
        a = x[:, self.idx_a]
        b = x[:, self.idx_b]

        # Truth-table values for every neuron, as bits (hard forward / soft gradient).
        c = ste_bit(self.weight)  # (out_dim, 4) = [f00, f01, f10, f11]
        f00, f01, f10, f11 = c[:, 0], c[:, 1], c[:, 2], c[:, 3]

        # Multilinear LUT evaluation, broadcast over the batch.
        return f00 + (f10 - f00) * a + (f01 - f00) * b + (f00 - f01 - f10 + f11) * (a * b)

    def extra_repr(self) -> str:
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}, fan_in=2"


# ======================================================================================
# 3. Thermometer encoder - turn real pixels into bits
# ======================================================================================


class Thermometer(nn.Module):
    """Encode each input feature as ``num_bits`` threshold bits ("thermometer code").

    For a value ``v`` and thresholds ``t_1 < t_2 < ... < t_k`` the code is
    ``[v > t_1, v > t_2, ..., v > t_k]`` - a unary-like representation that is monotone in
    ``v``. Thresholds are the per-channel *quantiles* of the fit data (a "distributive"
    thermometer), so each bit splits roughly the same number of pixels.

    Call :meth:`fit` once on a sample of training images before use. Input is ``(B, C, H, W)``
    and output is ``(B, C*num_bits, H, W)`` bits.
    """

    def __init__(self, num_bits: int = 2) -> None:
        super().__init__()
        self.num_bits = num_bits
        self.register_buffer("thresholds", None, persistent=True)  # (C, num_bits)

    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> "Thermometer":
        # x: (N, C, H, W). Compute per-channel quantile thresholds.
        c = x.shape[1]
        flat = x.permute(1, 0, 2, 3).reshape(c, -1)  # (C, N*H*W)
        qs = torch.linspace(0, 1, self.num_bits + 2)[1:-1]  # drop 0 and 1
        self.thresholds = torch.quantile(flat, qs.to(flat), dim=1).T.contiguous()  # (C, num_bits)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.thresholds is None:
            raise RuntimeError("Thermometer must be .fit() on data before use")
        # x: (B, C, H, W) -> compare against (1, C, 1, 1, num_bits)
        t = self.thresholds.view(1, x.shape[1], 1, 1, self.num_bits)
        bits = (x.unsqueeze(-1) > t).to(x.dtype)  # (B, C, H, W, num_bits)
        # Fold num_bits into the channel axis: (B, C*num_bits, H, W)
        b, c, h, w, k = bits.shape
        return bits.permute(0, 1, 4, 2, 3).reshape(b, c * k, h, w)


# ======================================================================================
# 4. GroupSum head - read bits out as class scores
# ======================================================================================


class GroupSum(nn.Module):
    """Split the final bit vector into ``k`` equal groups and sum each group (popcount).

    The last LUT layer has ``out_dim`` bits; we cut them into ``k`` classes of
    ``out_dim // k`` bits and count the ones per class. ``tau`` just scales the logits so
    the softmax/cross-entropy that follows is not over-confident (it has no effect at eval).
    """

    def __init__(self, k: int, tau: float = 1.0) -> None:
        super().__init__()
        self.k = k
        self.tau = tau

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % self.k != 0:
            raise ValueError(f"last dim {x.shape[-1]} not divisible by k={self.k}")
        groups = x.reshape(x.shape[0], self.k, x.shape[-1] // self.k)
        return groups.sum(-1) / self.tau

    def extra_repr(self) -> str:
        return f"k={self.k}, tau={self.tau}"


# ======================================================================================
# 5. The full model
# ======================================================================================


@dataclass
class Config:
    """All the knobs. Defaults are tuned to clearly OVERFIT a CIFAR-10 subset."""

    num_classes: int = 10
    in_channels: int = 3
    image_size: int = 32
    num_bits: int = 2          # thermometer bits per channel
    layer_widths: tuple[int, ...] = (12000, 12000, 12000)  # one LUTLayer per entry
    tau: float = 100.0         # head logit scale
    seed: int = 0


class LUTNet(nn.Sequential):
    """Thermometer -> Flatten -> N x LUTLayer -> GroupSum.

    Build it with a fitted thermometer so the input bit-width is known::

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
    """Convenience builder: fit a thermometer on ``fit_samples`` and assemble the model."""
    enc = Thermometer(num_bits=cfg.num_bits).fit(fit_samples)
    return LUTNet(cfg, encoder=enc)
