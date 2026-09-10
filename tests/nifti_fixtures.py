from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np


def make_task06(root: Path, case_count: int = 5, shape: tuple[int, int, int] = (16, 18, 12)) -> Path:
    for directory in ("imagesTr", "labelsTr", "imagesTs"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    affine = np.diag((-1.5, 1.0, 2.0, 1.0))
    training = []
    for index in range(case_count):
        case_id = f"lung_{index + 1:03d}"
        image = np.full(shape, -1000, dtype=np.float32)
        image[2:-2, 2:-2, 1:-1] = -600 + index * 10
        image[5:9, 6:10, 4:8] = 50 + index
        label = np.zeros(shape, dtype=np.uint8)
        extent = 1 + index % 3
        label[6 : 6 + extent, 7 : 7 + extent, 5 : 5 + extent] = 1
        image_path = root / "imagesTr" / f"{case_id}.nii.gz"
        label_path = root / "labelsTr" / f"{case_id}.nii.gz"
        nib.save(nib.Nifti1Image(image, affine), image_path)
        nib.save(nib.Nifti1Image(label, affine), label_path)
        training.append({"image": f"./imagesTr/{case_id}.nii.gz", "label": f"./labelsTr/{case_id}.nii.gz"})
    (root / "dataset.json").write_text(
        json.dumps({"name": "Task06_Lung synthetic", "training": training}), encoding="utf-8"
    )
    return root
