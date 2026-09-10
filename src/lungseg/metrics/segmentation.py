"""Per-volume binary segmentation metrics with declared empty-mask behavior."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import ndimage


def _binary(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array)
    if result.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {result.shape}")
    return result.astype(bool, copy=False)


def hd95(prediction: np.ndarray, target: np.ndarray, spacing: Sequence[float] = (1, 1, 1)) -> float:
    """Compute symmetric robust Hausdorff distance in physical units.

    Empty/empty is 0. A single empty mask yields positive infinity and is kept
    explicit so aggregators cannot silently discard failed detections.
    """
    prediction = _binary(prediction)
    target = _binary(target)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes must match")
    spacing = tuple(float(value) for value in spacing)
    if len(spacing) != 3 or any(value <= 0 for value in spacing):
        raise ValueError("spacing must contain three positive values")
    if not prediction.any() and not target.any():
        return 0.0
    if not prediction.any() or not target.any():
        return float("inf")

    structure = ndimage.generate_binary_structure(3, 1)
    prediction_surface = prediction ^ ndimage.binary_erosion(prediction, structure=structure)
    target_surface = target ^ ndimage.binary_erosion(target, structure=structure)
    target_distance = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    prediction_distance = ndimage.distance_transform_edt(~prediction_surface, sampling=spacing)
    distances = np.concatenate(
        (target_distance[prediction_surface], prediction_distance[target_surface])
    )
    return float(np.percentile(distances, 95))


def binary_case_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: Sequence[float] = (1, 1, 1),
) -> dict[str, float]:
    prediction = _binary(prediction)
    target = _binary(target)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes must match")

    true_positive = int(np.logical_and(prediction, target).sum())
    false_positive = int(np.logical_and(prediction, ~target).sum())
    false_negative = int(np.logical_and(~prediction, target).sum())
    prediction_count = int(prediction.sum())
    target_count = int(target.sum())
    union = int(np.logical_or(prediction, target).sum())

    both_empty = prediction_count == 0 and target_count == 0
    dice = 1.0 if both_empty else 2.0 * true_positive / (prediction_count + target_count)
    iou = 1.0 if both_empty else true_positive / union
    precision = 1.0 if prediction_count == 0 and target_count == 0 else (
        true_positive / prediction_count if prediction_count else 0.0
    )
    recall = 1.0 if target_count == 0 and prediction_count == 0 else (
        true_positive / target_count if target_count else 0.0
    )
    return {
        "dice": float(dice),
        "hd95": hd95(prediction, target, spacing),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "false_positive_voxels": float(false_positive),
        "false_negative_voxels": float(false_negative),
    }
