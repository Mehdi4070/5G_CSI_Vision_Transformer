import unittest

import torch

from src.models.cnn import CSILocalizationCNN


class CnnTests(unittest.TestCase):
    def test_forward_shape_and_finite_values(self) -> None:
        model = CSILocalizationCNN(channels=(8, 16), dropout=0.0)
        inputs = torch.randn(4, 3, 64, 64)

        outputs = model(inputs)

        self.assertEqual(tuple(outputs.shape), (4, 2))
        self.assertTrue(torch.isfinite(outputs).all())


if __name__ == "__main__":
    unittest.main()

