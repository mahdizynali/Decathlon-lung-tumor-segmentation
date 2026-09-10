from __future__ import annotations

import unittest

import numpy as np
import torch

from lungseg.losses import build_loss
from lungseg.metrics import binary_case_metrics, hd95


class LossMetric3DTests(unittest.TestCase):
    def test_dice_ce_backward_is_finite(self) -> None:
        logits = torch.randn(1, 2, 8, 8, 8, requires_grad=True)
        target = torch.zeros(1, 1, 8, 8, 8, dtype=torch.long)
        target[:, :, 2:5, 2:5, 2:5] = 1
        loss = build_loss()(logits, target)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)

    def test_perfect_and_disjoint_metrics(self) -> None:
        target = np.zeros((8, 8, 8), dtype=np.uint8)
        target[1:3, 1:3, 1:3] = 1
        perfect = binary_case_metrics(target, target, spacing=(2, 1, 1))
        self.assertEqual(perfect["dice"], 1.0)
        self.assertEqual(perfect["hd95"], 0.0)

        prediction = np.zeros_like(target)
        prediction[5:7, 5:7, 5:7] = 1
        disjoint = binary_case_metrics(prediction, target)
        self.assertEqual(disjoint["dice"], 0.0)
        self.assertGreater(disjoint["hd95"], 0.0)

    def test_empty_mask_policy(self) -> None:
        empty = np.zeros((4, 4, 4), dtype=np.uint8)
        foreground = empty.copy()
        foreground[1, 1, 1] = 1
        self.assertEqual(binary_case_metrics(empty, empty)["dice"], 1.0)
        self.assertEqual(hd95(empty, empty), 0.0)
        self.assertTrue(np.isinf(hd95(foreground, empty)))


if __name__ == "__main__":
    unittest.main()
