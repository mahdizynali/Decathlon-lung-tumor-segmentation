from __future__ import annotations

import unittest

import torch

from lungseg.models import RASegNet3D, build_model


class Model3DTests(unittest.TestCase):
    def test_ra_segnet_shape_and_attention_gradients(self) -> None:
        model = RASegNet3D(channels=(4, 8, 16), normalization="GROUP")
        inputs = torch.randn(1, 1, 17, 19, 21)
        output = model(inputs)
        output.mean().backward()

        self.assertEqual(tuple(output.shape), (1, 2, 17, 19, 21))
        attention_gradients = [
            parameter.grad for name, parameter in model.named_parameters() if name.startswith("attention")
        ]
        self.assertTrue(attention_gradients)
        self.assertTrue(all(gradient is not None for gradient in attention_gradients))
        for block in model.encoder:
            self.assertIsNotNone(block.conv1.weight.grad)

    def test_registry_builds_monai_unet_and_segresnet(self) -> None:
        unet = build_model(
            {"name": "unet3d", "in_channels": 1, "out_channels": 2, "channels": [4, 8, 16], "strides": [2, 2], "normalization": "INSTANCE"}
        )
        segresnet = build_model(
            {"name": "segresnet", "in_channels": 1, "out_channels": 2, "init_filters": 4, "normalization": "INSTANCE"}
        )
        inputs = torch.randn(1, 1, 16, 16, 16)
        self.assertEqual(tuple(unet(inputs).shape), (1, 2, 16, 16, 16))
        self.assertEqual(tuple(segresnet(inputs).shape), (1, 2, 16, 16, 16))


if __name__ == "__main__":
    unittest.main()
