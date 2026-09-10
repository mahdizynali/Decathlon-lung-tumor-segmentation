"""Traceable MONAI transforms for volumetric CT segmentation."""

from __future__ import annotations

from collections.abc import Sequence

from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    ScaleIntensityRanged,
    Spacingd,
)


def _triple(values: Sequence[float] | None, name: str) -> tuple[float, float, float]:
    if values is None or len(values) != 3:
        raise ValueError(f"{name} must contain three audit-selected values")
    result = tuple(float(value) for value in values)
    if any(value <= 0 for value in result):
        raise ValueError(f"{name} values must be positive")
    return result


def _window(lower_hu: float | None, upper_hu: float | None) -> tuple[float, float]:
    if lower_hu is None or upper_hu is None:
        raise ValueError("CT window is unset; select lower_hu and upper_hu from the training-fold audit")
    if lower_hu >= upper_hu:
        raise ValueError("lower_hu must be less than upper_hu")
    return float(lower_hu), float(upper_hu)


def deterministic_transforms(
    target_spacing: Sequence[float],
    lower_hu: float,
    upper_hu: float,
    orientation: str = "RAS",
) -> Compose:
    """Load, orient, resample, window, and crop an image/label pair."""
    spacing = _triple(target_spacing, "target_spacing")
    lower, upper = _window(lower_hu, upper_hu)
    keys = ("image", "label")
    return Compose(
        [
            LoadImaged(keys=keys, image_only=False),
            EnsureChannelFirstd(keys=keys),
            Orientationd(
                keys=keys,
                axcodes=orientation,
                labels=(("L", "R"), ("P", "A"), ("I", "S")),
            ),
            Spacingd(keys=keys, pixdim=spacing, mode=("bilinear", "nearest")),
            ScaleIntensityRanged(
                keys="image", a_min=lower, a_max=upper, b_min=0.0, b_max=1.0, clip=True
            ),
            CropForegroundd(keys=keys, source_key="image", select_fn=lambda image: image > 0),
            EnsureTyped(keys=keys, dtype=None, track_meta=True),
        ]
    )


def training_transforms(
    target_spacing: Sequence[float],
    lower_hu: float,
    upper_hu: float,
    patch_size: Sequence[int],
    *,
    samples_per_volume: int = 2,
    positive_weight: float = 1.0,
    negative_weight: float = 1.0,
    orientation: str = "RAS",
) -> Compose:
    """Build deterministic preprocessing plus online paired 3D augmentation."""
    roi = tuple(int(value) for value in _triple(patch_size, "patch_size"))
    if samples_per_volume < 1:
        raise ValueError("samples_per_volume must be positive")
    if positive_weight < 0 or negative_weight < 0 or positive_weight + negative_weight == 0:
        raise ValueError("positive/negative sampling weights must be non-negative and not both zero")

    deterministic = deterministic_transforms(target_spacing, lower_hu, upper_hu, orientation)
    return Compose(
        [
            *deterministic.transforms,
            RandCropByPosNegLabeld(
                keys=("image", "label"),
                label_key="label",
                spatial_size=roi,
                pos=positive_weight,
                neg=negative_weight,
                num_samples=samples_per_volume,
                image_key="image",
                image_threshold=0.0,
                allow_smaller=False,
            ),
            RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=0),
            RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=1),
            RandAffined(
                keys=("image", "label"),
                prob=0.2,
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1),
                mode=("bilinear", "nearest"),
                padding_mode="border",
            ),
            RandGaussianNoised(keys="image", prob=0.15, mean=0.0, std=0.01),
            EnsureTyped(keys=("image", "label"), dtype=None, track_meta=True),
        ]
    )
