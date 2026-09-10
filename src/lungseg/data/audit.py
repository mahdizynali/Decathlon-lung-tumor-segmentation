"""NIfTI integrity checks and per-case Task06 Lung statistics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
from scipy import ndimage

from lungseg.data.discovery import LungCase


@dataclass(frozen=True)
class CaseAudit:
    case_id: str
    shape: tuple[int, ...]
    spacing_mm: tuple[float, ...]
    orientation: tuple[str, ...]
    affine_determinant: float
    finite_intensities: bool
    intensity_min: float
    intensity_max: float
    intensity_percentiles: dict[str, float]
    label_values: list[int]
    tumor_voxels: int
    tumor_volume_mm3: float
    foreground_percent: float
    connected_components: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _valid_affine(affine: np.ndarray, case_id: str) -> float:
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError(f"{case_id}: affine must be a finite 4x4 matrix")
    determinant = float(np.linalg.det(affine[:3, :3]))
    if abs(determinant) < 1e-8:
        raise ValueError(f"{case_id}: affine is singular")
    return determinant


def audit_case(
    case: LungCase,
    percentiles: Sequence[float] = (0.5, 1.0, 50.0, 99.0, 99.5),
) -> CaseAudit:
    """Validate one labeled case and return JSON-safe quantitative metadata."""
    if case.label is None:
        raise ValueError(f"{case.case_id}: auditing tumor statistics requires a label")
    image_nifti = nib.load(case.image)
    label_nifti = nib.load(case.label)
    image = np.asanyarray(image_nifti.dataobj, dtype=np.float32)
    label = np.asanyarray(label_nifti.dataobj)

    if image.ndim != 3 or label.ndim != 3:
        raise ValueError(f"{case.case_id}: expected 3D image and label")
    if image.shape != label.shape:
        raise ValueError(f"{case.case_id}: image shape {image.shape} != label shape {label.shape}")
    if not np.allclose(image_nifti.affine, label_nifti.affine, rtol=1e-5, atol=1e-4):
        raise ValueError(f"{case.case_id}: image and label affines do not match")
    determinant = _valid_affine(image_nifti.affine, case.case_id)
    finite = bool(np.isfinite(image).all())
    if not finite:
        raise ValueError(f"{case.case_id}: image contains NaN or infinite values")
    if not np.isfinite(label).all() or not np.allclose(label, np.rint(label)):
        raise ValueError(f"{case.case_id}: label must contain finite integer values")

    label = label.astype(np.int64, copy=False)
    unique = np.unique(label)
    if np.any(unique < 0):
        raise ValueError(f"{case.case_id}: label contains negative classes: {unique.tolist()}")
    tumor = label > 0
    tumor_voxels = int(tumor.sum())
    spacing = tuple(float(value) for value in image_nifti.header.get_zooms()[:3])
    voxel_volume = float(abs(determinant))
    components = int(ndimage.label(tumor)[1]) if tumor_voxels else 0
    percentile_values = np.percentile(image, percentiles)
    percentile_map = {f"p{value:g}": float(result) for value, result in zip(percentiles, percentile_values)}

    return CaseAudit(
        case_id=case.case_id,
        shape=tuple(int(value) for value in image.shape),
        spacing_mm=spacing,
        orientation=tuple(str(value) for value in nib.aff2axcodes(image_nifti.affine)),
        affine_determinant=determinant,
        finite_intensities=finite,
        intensity_min=float(image.min()),
        intensity_max=float(image.max()),
        intensity_percentiles=percentile_map,
        label_values=[int(value) for value in unique],
        tumor_voxels=tumor_voxels,
        tumor_volume_mm3=tumor_voxels * voxel_volume,
        foreground_percent=100.0 * tumor_voxels / image.size,
        connected_components=components,
    )


def summarize_audits(audits: Sequence[CaseAudit]) -> dict[str, Any]:
    if not audits:
        raise ValueError("Cannot summarize an empty audit")
    spacings = np.asarray([audit.spacing_mm for audit in audits], dtype=np.float64)
    tumor_volumes = np.asarray([audit.tumor_volume_mm3 for audit in audits], dtype=np.float64)
    return {
        "case_count": len(audits),
        "empty_mask_count": sum(audit.tumor_voxels == 0 for audit in audits),
        "spacing_mm": {
            "minimum": spacings.min(axis=0).tolist(),
            "median": np.median(spacings, axis=0).tolist(),
            "maximum": spacings.max(axis=0).tolist(),
        },
        "tumor_volume_mm3": {
            "minimum": float(tumor_volumes.min()),
            "median": float(np.median(tumor_volumes)),
            "maximum": float(tumor_volumes.max()),
        },
    }
