from __future__ import annotations

import random
import unittest

import numpy as np

from lungseg.utils.reproducibility import seed_everything


class ReproducibilityTests(unittest.TestCase):
    def test_reseeding_repeats_python_numpy_and_torch_values(self) -> None:
        seed_everything(43)
        first_python = random.random()
        first_numpy = np.random.random()

        seed_everything(43)
        self.assertEqual(random.random(), first_python)
        self.assertEqual(np.random.random(), first_numpy)

        try:
            import torch
        except ImportError:
            return

        seed_everything(43)
        first_torch = torch.rand(3)
        seed_everything(43)
        self.assertTrue(torch.equal(torch.rand(3), first_torch))

    def test_negative_seed_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            seed_everything(-1)


if __name__ == "__main__":
    unittest.main()
