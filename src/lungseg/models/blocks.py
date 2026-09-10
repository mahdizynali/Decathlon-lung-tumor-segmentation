"""Reusable residual and attention components for 3D segmentation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def normalization(channels: int, kind: str = "INSTANCE") -> nn.Module:
    kind = kind.upper()
    if kind == "INSTANCE":
        return nn.InstanceNorm3d(channels, affine=True)
    if kind == "GROUP":
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unsupported 3D normalization: {kind}")


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str = "INSTANCE") -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = normalization(out_channels, norm)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = normalization(out_channels, norm)
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.projection = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.projection(inputs)
        output = self.activation(self.norm1(self.conv1(inputs)))
        output = self.norm2(self.conv2(output))
        return self.activation(output + residual)


class AttentionGate3D(nn.Module):
    """Gate one encoder skip using the corresponding decoder feature map."""

    def __init__(self, skip_channels: int, gate_channels: int, intermediate_channels: int) -> None:
        super().__init__()
        self.skip_projection = nn.Conv3d(skip_channels, intermediate_channels, kernel_size=1, bias=False)
        self.gate_projection = nn.Conv3d(gate_channels, intermediate_channels, kernel_size=1, bias=False)
        self.score = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(intermediate_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        gate = F.interpolate(gate, size=skip.shape[2:], mode="trilinear", align_corners=False)
        weights = self.score(self.skip_projection(skip) + self.gate_projection(gate))
        return skip * weights
