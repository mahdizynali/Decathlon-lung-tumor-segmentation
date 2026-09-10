"""Construct supported 3D models from explicit configuration mappings."""

from __future__ import annotations

from typing import Any, Mapping

from monai.networks.nets import SegResNet, UNet
from torch import nn

from lungseg.models.ra_segnet3d import RASegNet3D


def build_model(config: Mapping[str, Any]) -> nn.Module:
    model = config.get("model", config)
    name = str(model.get("name", "")).lower()
    in_channels = int(model.get("in_channels", 1))
    out_channels = int(model.get("out_channels", 2))
    norm = str(model.get("normalization", "INSTANCE")).upper()

    if name == "unet3d":
        return UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=tuple(int(value) for value in model["channels"]),
            strides=tuple(int(value) for value in model["strides"]),
            num_res_units=int(model.get("num_res_units", 0)),
            norm=norm,
        )
    if name == "segresnet":
        return SegResNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=int(model.get("init_filters", 32)),
            norm=norm,
        )
    if name == "ra_segnet3d":
        return RASegNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            channels=tuple(int(value) for value in model["channels"]),
            normalization=norm,
        )
    raise ValueError(f"Unknown model name: {name!r}")
