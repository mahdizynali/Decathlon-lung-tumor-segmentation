"""Minimal residual-attention 3D encoder-decoder with explicit skip flow."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from lungseg.models.blocks import AttentionGate3D, ResidualBlock3D


class RASegNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        channels: Sequence[int] = (32, 64, 128, 256),
        normalization: str = "INSTANCE",
    ) -> None:
        super().__init__()
        if len(channels) < 3:
            raise ValueError("RA-SegNet3D requires at least three channel stages")
        if any(channel < 1 for channel in channels):
            raise ValueError("All channel counts must be positive")

        widths = tuple(int(channel) for channel in channels)
        self.encoder = nn.ModuleList()
        previous = in_channels
        for width in widths:
            self.encoder.append(ResidualBlock3D(previous, width, normalization))
            previous = width
        self.downsample = nn.MaxPool3d(kernel_size=2, stride=2)

        self.upconvs = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.decoder = nn.ModuleList()
        current = widths[-1]
        for skip_width in reversed(widths[:-1]):
            self.upconvs.append(nn.ConvTranspose3d(current, skip_width, kernel_size=2, stride=2))
            self.attention.append(
                AttentionGate3D(skip_width, skip_width, max(1, skip_width // 2))
            )
            self.decoder.append(ResidualBlock3D(skip_width * 2, skip_width, normalization))
            current = skip_width
        self.head = nn.Conv3d(widths[0], out_channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        input_shape = inputs.shape[2:]
        skips: list[torch.Tensor] = []
        output = inputs
        for index, block in enumerate(self.encoder):
            output = block(output)
            if index < len(self.encoder) - 1:
                skips.append(output)
                output = self.downsample(output)

        for upconv, attention, decoder, skip in zip(
            self.upconvs, self.attention, self.decoder, reversed(skips)
        ):
            output = upconv(output)
            if output.shape[2:] != skip.shape[2:]:
                output = F.interpolate(output, size=skip.shape[2:], mode="trilinear", align_corners=False)
            gated_skip = attention(skip, output)
            output = decoder(torch.cat((output, gated_skip), dim=1))

        logits = self.head(output)
        if logits.shape[2:] != input_shape:
            logits = F.interpolate(logits, size=input_shape, mode="trilinear", align_corners=False)
        return logits
