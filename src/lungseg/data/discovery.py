"""Discover MSD Task06 Lung volumes without relying on directory order."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


NIFTI_SUFFIX = ".nii.gz"


@dataclass(frozen=True)
class LungCase:
    """Paths belonging to one volumetric CT case."""

    case_id: str
    image: Path
    label: Path | None
    labeled: bool

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["image"] = str(self.image)
        value["label"] = str(self.label) if self.label is not None else None
        return value

    def as_monai_dict(self) -> dict[str, str]:
        item = {"case_id": self.case_id, "image": str(self.image)}
        if self.label is not None:
            item["label"] = str(self.label)
        return item


def case_id_from_path(path: Path) -> str:
    name = path.name
    if not name.endswith(NIFTI_SUFFIX):
        raise ValueError(f"Expected a .nii.gz file, got: {path}")
    return name[: -len(NIFTI_SUFFIX)]


def _index_nifti(directory: Path, *, required: bool) -> dict[str, Path]:
    if not directory.is_dir():
        if required:
            raise FileNotFoundError(f"Required dataset directory is missing: {directory}")
        return {}
    indexed: dict[str, Path] = {}
    for path in sorted(directory.glob(f"*{NIFTI_SUFFIX}")):
        case_id = case_id_from_path(path)
        if case_id in indexed:
            raise ValueError(f"Duplicate case ID {case_id!r} in {directory}")
        indexed[case_id] = path.resolve()
    return indexed


def discover_cases(
    root: Path,
    *,
    include_unlabeled_test: bool = False,
    require_dataset_json: bool = True,
) -> list[LungCase]:
    """Discover labeled training cases and optionally official unlabeled cases."""
    root = Path(root).expanduser().resolve()
    dataset_json = root / "dataset.json"
    if require_dataset_json and not dataset_json.is_file():
        raise FileNotFoundError(f"Required dataset metadata is missing: {dataset_json}")
    if dataset_json.is_file():
        with dataset_json.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        if not isinstance(metadata, dict):
            raise ValueError(f"dataset.json root must be an object: {dataset_json}")

    images = _index_nifti(root / "imagesTr", required=True)
    labels = _index_nifti(root / "labelsTr", required=True)
    if not images:
        raise ValueError(f"No training NIfTI images found under {root / 'imagesTr'}")

    missing_labels = sorted(images.keys() - labels.keys())
    orphan_labels = sorted(labels.keys() - images.keys())
    if missing_labels or orphan_labels:
        raise ValueError(
            "Image/label IDs do not match; "
            f"missing labels={missing_labels}, orphan labels={orphan_labels}"
        )

    cases = [LungCase(case_id, images[case_id], labels[case_id], True) for case_id in sorted(images)]
    if include_unlabeled_test:
        tests = _index_nifti(root / "imagesTs", required=False)
        duplicate_ids = sorted(images.keys() & tests.keys())
        if duplicate_ids:
            raise ValueError(f"Training and test IDs overlap: {duplicate_ids}")
        cases.extend(LungCase(case_id, tests[case_id], None, False) for case_id in sorted(tests))
    return cases


def ensure_unique_case_ids(cases: Iterable[LungCase]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for case in cases:
        if case.case_id in seen:
            duplicates.add(case.case_id)
        seen.add(case.case_id)
    if duplicates:
        raise ValueError(f"Duplicate case IDs: {sorted(duplicates)}")
