import unittest

import torch

from src.models.hybrid import CSILocalizationHybrid


class HybridTests(unittest.TestCase):
    def test_forward_shape_tokens_and_finite_values(self) -> None:
        model = CSILocalizationHybrid(
            image_size=(64, 64),
            stem_channels=(8, 16, 32),
            embedding_dim=32,
            depth=2,
            num_heads=4,
            dropout=0.0,
        )
        outputs = model(torch.randn(4, 3, 64, 64))

        self.assertEqual(tuple(outputs.shape), (4, 2))
        self.assertTrue(torch.isfinite(outputs).all())
        self.assertEqual(model.patch_grid, (8, 8))
        self.assertEqual(tuple(model.antenna_position.shape), (1, 8, 1, 32))
        self.assertEqual(tuple(model.subcarrier_position.shape), (1, 1, 8, 32))

    def test_rejects_incompatible_image_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            CSILocalizationHybrid(image_size=(62, 64), stem_channels=(8, 16, 32))


if __name__ == "__main__":
    unittest.main()
