"""Deterministic patient-level cross-validation split generation."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from lungseg.data.audit import CaseAudit


def dataset_fingerprint(audits: Sequence[CaseAudit]) -> str:
    records = [
        {"case_id": item.case_id, "shape": item.shape, "spacing_mm": item.spacing_mm, "tumor_voxels": item.tumor_voxels}
        for item in sorted(audits, key=lambda value: value.case_id)
    ]
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode("utf-8")).hexdigest()


def _volume_bins(audits: Sequence[CaseAudit], folds: int) -> np.ndarray | None:
    volumes = np.asarray([audit.tumor_volume_mm3 for audit in audits], dtype=np.float64)
    if np.unique(volumes).size < 2:
        return None
    max_bins = min(4, len(audits) // folds)
    for bin_count in range(max_bins, 1, -1):
        edges = np.unique(np.quantile(volumes, np.linspace(0, 1, bin_count + 1)))
        if edges.size < 3:
            continue
        bins = np.digitize(volumes, edges[1:-1], right=True)
        counts = np.bincount(bins)
        if counts.size == bin_count and counts.min() >= folds:
            return bins
    return None


def make_folds(audits: Sequence[CaseAudit], folds: int = 5, seed: int = 43) -> dict[str, Any]:
    """Create complete non-overlapping folds, stratified by tumor volume when safe."""
    ordered = sorted(audits, key=lambda value: value.case_id)
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if len(ordered) < folds:
        raise ValueError(f"Need at least {folds} cases, got {len(ordered)}")
    ids = np.asarray([audit.case_id for audit in ordered])
    bins = _volume_bins(ordered, folds)
    if bins is None:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        iterator = splitter.split(ids)
        method = "seeded_kfold"
    else:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        iterator = splitter.split(ids, bins)
        method = "tumor_volume_stratified_kfold"

    fold_records = []
    for index, (train_indices, validation_indices) in enumerate(iterator):
        fold_records.append(
            {
                "fold": index,
                "train": ids[train_indices].tolist(),
                "validation": ids[validation_indices].tolist(),
            }
        )
    result = {
        "dataset": "MSD_Task06_Lung",
        "seed": seed,
        "num_folds": folds,
        "method": method,
        "dataset_fingerprint": dataset_fingerprint(ordered),
        "folds": fold_records,
    }
    validate_folds(result, set(ids.tolist()))
    return result


def validate_folds(split: dict[str, Any], expected_ids: set[str]) -> None:
    validation_ids: list[str] = []
    for fold in split.get("folds", []):
        train = set(fold["train"])
        validation = set(fold["validation"])
        if train & validation:
            raise ValueError(f"Fold {fold['fold']} has train/validation leakage")
        if train | validation != expected_ids:
            raise ValueError(f"Fold {fold['fold']} does not cover the complete dataset")
        validation_ids.extend(fold["validation"])
    if len(validation_ids) != len(expected_ids) or set(validation_ids) != expected_ids:
        raise ValueError("Every case must appear in validation exactly once")
