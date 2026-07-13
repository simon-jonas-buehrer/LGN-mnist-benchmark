"""FIXED (non-learned) image featurization for the NAND net's INPUT ENCODING.

The wiring search was capped near ~30% because its only inputs were thermometer bits of RAW pixels
-- it had to rediscover every spatial relationship from random wiring. A LINEAR probe jumps from
29.5% to 42.8% val the moment horizontal+vertical edge differences are added (see feat_probe.py),
so feeding the search those edges/gradients directly is the single cheapest lever.

IMPORTANT (scope): these are FIXED local differences/statistics applied ONCE as input encoding, with
NO learned weights and NO weight-shared conv LAYER in the network -- the NAND net itself stays a
plain fully-connected FFN. This is feature engineering on the input (like HOG/thermometer), not a
convolutional architecture.

expand(x, families) -> (B, Cf, H, W) float feature maps; the caller thermometer-encodes them to bits.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

# default families: raw pixels + 4 edge orientations + Laplacian + colour-opponent. Each is a fixed
# local operator; together a linear probe reaches ~46% (vs 29.5% raw) -- headroom the search exploits.
DEFAULT = ("raw", "gh", "gv", "gd", "lap", "oppo")


def _shift_diff(x, dy, dx):
    xp = F.pad(x, (1, 1, 1, 1), mode="reflect")
    return x - xp[:, :, 1 + dy:1 + dy + x.shape[2], 1 + dx:1 + dx + x.shape[3]]


def expand(x: torch.Tensor, families=DEFAULT) -> torch.Tensor:
    """(B,3,H,W) image -> (B, Cf, H, W) fixed feature maps for the requested families, concatenated."""
    outs = []
    for fam in families:
        if fam == "raw":
            outs.append(x)
        elif fam == "gh":
            outs.append(_shift_diff(x, 0, 1))
        elif fam == "gv":
            outs.append(_shift_diff(x, 1, 0))
        elif fam == "gd":
            outs.append(_shift_diff(x, 1, 1)); outs.append(_shift_diff(x, 1, -1))
        elif fam == "lap":
            xp = F.pad(x, (1, 1, 1, 1), mode="reflect")
            outs.append(4 * x - xp[:, :, :-2, 1:-1] - xp[:, :, 2:, 1:-1]
                        - xp[:, :, 1:-1, :-2] - xp[:, :, 1:-1, 2:])
        elif fam == "blur":
            outs.append(F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3, 1))
        elif fam == "oppo":
            r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            outs.append(torch.cat([r - g, b - (r + g) / 2], 1))
        else:
            raise ValueError(f"unknown feature family {fam!r}")
    return torch.cat(outs, 1)


def n_channels(families=DEFAULT) -> int:
    per = {"raw": 3, "gh": 3, "gv": 3, "gd": 6, "lap": 3, "blur": 3, "oppo": 2}
    return sum(per[f] for f in families)
