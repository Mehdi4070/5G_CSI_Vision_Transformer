import unittest

import torch

from src.models.vit import CSILocalizationViT


class ViTTests(unittest.TestCase):
    def test_forward_shape_and_finite_values(self) -> None:
        model = CSILocalizationViT(
            image_size=(64, 64),
            patch_size=(8, 8),
            embedding_dim=32,
            depth=2,
            num_heads=4,
            dropout=0.0,
        )
        outputs = model(torch.randn(4, 3, 64, 64))

        self.assertEqual(tuple(outputs.shape), (4, 2))
        self.assertTrue(torch.isfinite(outputs).all())
        self.assertEqual(tuple(model.antenna_position.shape), (1, 8, 1, 32))
        self.assertEqual(tuple(model.subcarrier_position.shape), (1, 1, 8, 32))

    def test_rejects_incompatible_patch_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            CSILocalizationViT(image_size=(64, 64), patch_size=(7, 8))


if __name__ == "__main__":
    unittest.main()
