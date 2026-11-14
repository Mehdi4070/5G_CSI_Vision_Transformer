import unittest

import numpy as np

from src.metrics import localization_errors, localization_metrics, mean_position_baseline


class LocalizationMetricsTests(unittest.TestCase):
    def test_known_euclidean_errors(self) -> None:
        predictions = np.array([[0.0, 0.0], [0.0, 0.0]])
        targets = np.array([[0.0, 0.0], [3.0, 4.0]])

        errors = localization_errors(predictions, targets)
        metrics = localization_metrics(predictions, targets)

        np.testing.assert_allclose(errors, [0.0, 5.0])
        self.assertEqual(metrics["mean_error_m"], 2.5)
        self.assertEqual(metrics["median_error_m"], 2.5)
        self.assertAlmostEqual(metrics["rmse_m"], np.sqrt(12.5))

    def test_mean_position_baseline(self) -> None:
        train = np.array([[0.0, 0.0], [2.0, 4.0]])
        evaluation = np.array([[1.0, 2.0], [4.0, 6.0]])

        predictions, metrics = mean_position_baseline(train, evaluation)

        np.testing.assert_allclose(predictions, [[1.0, 2.0], [1.0, 2.0]])
        self.assertEqual(metrics["num_users"], 2)


if __name__ == "__main__":
    unittest.main()

