"""Sliding-window whole-volume inference and geometry-preserving output."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from torch import nn


@torch.inference_mode()
def predict_volume(
    model: nn.Module,
    image: torch.Tensor,
    roi_size: Sequence[int],
    *,
    overlap: float = 0.5,
    sliding_window_batch_size: int = 1,
) -> torch.Tensor:
    """Return a discrete whole-volume prediction for ``[B, C, H, W, D]`` input."""
    if image.ndim != 5:
        raise ValueError(f"Expected [B,C,H,W,D] image, got shape {tuple(image.shape)}")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    model.eval()
    logits = sliding_window_inference(
        inputs=image,
        roi_size=tuple(int(value) for value in roi_size),
        sw_batch_size=sliding_window_batch_size,
        predictor=model,
        overlap=overlap,
        mode="gaussian",
    )
    if logits.shape[1] == 1:
        return (torch.sigmoid(logits) >= 0.5).to(torch.uint8)
    return torch.argmax(logits, dim=1, keepdim=True).to(torch.uint8)


def save_prediction(prediction: np.ndarray | torch.Tensor, reference: Path, output: Path) -> Path:
    """Save a 3D label map with the original reference affine and header."""
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    array = np.asarray(prediction).squeeze()
    if array.ndim != 3:
        raise ValueError(f"Expected a squeezable 3D prediction, got shape {np.asarray(prediction).shape}")
    reference_nifti = nib.load(Path(reference))
    if array.shape != reference_nifti.shape:
        raise ValueError(
            f"Prediction shape {array.shape} does not match reference shape {reference_nifti.shape}; "
            "invert preprocessing before saving"
        )
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = reference_nifti.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(array.astype(np.uint8), reference_nifti.affine, header), destination)
    return destination
