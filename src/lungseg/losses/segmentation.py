"""Well-tested MONAI losses for small-target segmentation."""

from __future__ import annotations

from monai.losses import DiceCELoss
from torch import nn


def build_loss(name: str = "dice_ce", include_background: bool = False) -> nn.Module:
    if name.lower() != "dice_ce":
        raise ValueError(f"Unsupported loss: {name!r}")
    return DiceCELoss(
        include_background=include_background,
        to_onehot_y=True,
        softmax=True,
        squared_pred=False,
        smooth_nr=1e-5,
        smooth_dr=1e-5,
    )
