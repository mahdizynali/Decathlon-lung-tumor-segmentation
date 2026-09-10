"""MONAI dataset and DataLoader construction for volumetric cases."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from monai.data import DataLoader, Dataset
from monai.transforms import Transform

from lungseg.data.discovery import LungCase, ensure_unique_case_ids


def make_dataset(cases: Sequence[LungCase], transform: Transform | None = None) -> Dataset:
    ensure_unique_case_ids(cases)
    if not cases:
        raise ValueError("At least one case is required")
    return Dataset(data=[case.as_monai_dict() for case in cases], transform=transform)


def make_loader(
    cases: Sequence[LungCase],
    transform: Transform,
    *,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    **kwargs: Any,
) -> DataLoader:
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    return DataLoader(
        make_dataset(cases, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs,
    )
