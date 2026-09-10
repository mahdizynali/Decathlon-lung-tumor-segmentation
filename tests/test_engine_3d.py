from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch import nn

from lungseg.engine.inference import predict_volume, save_prediction
from lungseg.engine.trainer import train_epoch
from lungseg.losses import build_loss


class ThresholdModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cat((-inputs, inputs), dim=1)


class Engine3DTests(unittest.TestCase):
    def test_sliding_window_prediction_and_nifti_geometry(self) -> None:
        image = torch.full((1, 1, 12, 14, 10), -1.0)
        image[:, :, 3:7, 4:9, 2:6] = 1.0
        prediction = predict_volume(ThresholdModel(), image, (8, 8, 8), overlap=0.25)
        self.assertEqual(tuple(prediction.shape), (1, 1, 12, 14, 10))
        self.assertEqual(int(prediction.sum()), 4 * 5 * 4)

        with tempfile.TemporaryDirectory() as directory:
            affine = np.diag((1.5, 1.0, 2.0, 1.0))
            reference = Path(directory) / "image.nii.gz"
            output = Path(directory) / "prediction.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros((12, 14, 10), dtype=np.float32), affine), reference)
            save_prediction(prediction, reference, output)
            saved = nib.load(output)
            self.assertEqual(saved.shape, (12, 14, 10))
            np.testing.assert_allclose(saved.affine, affine)

    def test_cpu_training_step_updates_model(self) -> None:
        model = nn.Conv3d(1, 2, kernel_size=1)
        before = model.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        batch = {
            "image": torch.randn(1, 1, 8, 8, 8),
            "label": torch.randint(0, 2, (1, 1, 8, 8, 8)),
        }
        loss = train_epoch(model, [batch], build_loss(), optimizer, torch.device("cpu"), amp=False)
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(before, model.weight.detach()))


if __name__ == "__main__":
    unittest.main()
