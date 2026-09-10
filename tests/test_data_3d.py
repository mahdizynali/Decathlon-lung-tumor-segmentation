from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from lungseg.data.audit import audit_case, summarize_audits
from lungseg.data.discovery import discover_cases
from lungseg.data.splits import make_folds
from lungseg.data.transforms import deterministic_transforms, training_transforms
from nifti_fixtures import make_task06


class Data3DTests(unittest.TestCase):
    def test_discovery_audit_and_folds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = make_task06(Path(directory) / "Task06_Lung")
            cases = discover_cases(root)
            audits = [audit_case(case) for case in cases]
            first = audits[0]

            self.assertEqual([case.case_id for case in cases], sorted(case.case_id for case in cases))
            self.assertEqual(first.shape, (16, 18, 12))
            self.assertEqual(first.orientation, ("L", "A", "S"))
            self.assertEqual(first.spacing_mm, (1.5, 1.0, 2.0))
            self.assertGreater(first.tumor_volume_mm3, 0)
            self.assertEqual(summarize_audits(audits)["case_count"], 5)

            split_a = make_folds(audits, folds=5, seed=43)
            split_b = make_folds(list(reversed(audits)), folds=5, seed=43)
            self.assertEqual(split_a, split_b)
            validation = [case for fold in split_a["folds"] for case in fold["validation"]]
            self.assertCountEqual(validation, [case.case_id for case in cases])

    def test_discovery_rejects_missing_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = make_task06(Path(directory) / "Task06_Lung", case_count=2)
            next((root / "labelsTr").glob("*.nii.gz")).unlink()
            with self.assertRaisesRegex(ValueError, "IDs do not match"):
                discover_cases(root)

    def test_transforms_align_image_and_discrete_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case = discover_cases(make_task06(Path(directory) / "Task06_Lung", case_count=1))[0]
            transform = deterministic_transforms((1.5, 1.0, 2.0), -1000, 400)
            result = transform(case.as_monai_dict())

            self.assertEqual(result["image"].shape, result["label"].shape)
            self.assertTrue(set(np.unique(result["label"]).tolist()) <= {0.0, 1.0})
            self.assertGreaterEqual(float(result["image"].min()), 0.0)
            self.assertLessEqual(float(result["image"].max()), 1.0)

            train = training_transforms(
                (1.5, 1.0, 2.0), -1000, 400, (8, 8, 8), samples_per_volume=1
            )(case.as_monai_dict())
            self.assertEqual(len(train), 1)
            self.assertEqual(tuple(train[0]["image"].shape), (1, 8, 8, 8))
            self.assertTrue(set(np.unique(train[0]["label"]).tolist()) <= {0.0, 1.0})

    def test_transform_requires_audit_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "CT window is unset"):
            deterministic_transforms((1, 1, 1), None, None)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
