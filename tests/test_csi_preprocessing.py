import unittest

import numpy as np

from src.csi_preprocessing import (
    csi_to_features,
    fit_feature_standardizer,
    standardize_features,
)


class CsiPreprocessingTests(unittest.TestCase):
    def test_amplitude_and_continuous_phase_channels(self) -> None:
        csi = np.array([[[1 + 0j, 0 + 1j, -1 + 0j]]], dtype=np.complex64)

        features = csi_to_features(csi, amplitude_floor=1e-12)

        np.testing.assert_allclose(features[0, 0], 0.0, atol=1e-6)
        np.testing.assert_allclose(features[0, 1, 0], [1.0, 0.0, -1.0], atol=1e-6)
        np.testing.assert_allclose(features[0, 2, 0], [0.0, 1.0, 0.0], atol=1e-6)

    def test_statistics_use_only_training_indices(self) -> None:
        rng = np.random.default_rng(7)
        csi = (rng.normal(size=(5, 2, 3)) + 1j * rng.normal(size=(5, 2, 3))).astype(
            np.complex64
        )
        train_indices = np.array([0, 1, 2, 3])
        stats = fit_feature_standardizer(
            csi, train_indices, amplitude_floor=1e-12, chunk_size=2
        )
        train_features = csi_to_features(csi[train_indices], amplitude_floor=1e-12)
        normalized = standardize_features(
            train_features, mean=np.array(stats["mean"]), std=np.array(stats["std"])
        )

        np.testing.assert_allclose(normalized.mean(axis=(0, 2, 3)), 0.0, atol=1e-5)
        np.testing.assert_allclose(normalized.std(axis=(0, 2, 3)), 1.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()

