import unittest

import numpy as np

from src.csi_dimensions import (
    centered_subarray_indices,
    select_csi_dimensions,
    uniform_subcarrier_indices,
)


class CSIDimensionSelectionTests(unittest.TestCase):
    def test_centered_physical_subarrays_have_requested_sizes(self) -> None:
        indices_16 = centered_subarray_indices((8, 8), (4, 4))
        indices_32 = centered_subarray_indices((8, 8), (4, 8))
        np.testing.assert_array_equal(
            indices_16,
            np.array([18, 19, 20, 21, 26, 27, 28, 29, 34, 35, 36, 37, 42, 43, 44, 45]),
        )
        self.assertEqual(len(indices_32), 32)
        self.assertEqual(len(np.unique(indices_32)), 32)

    def test_subcarriers_cover_the_full_band(self) -> None:
        indices = uniform_subcarrier_indices(64, 16)
        self.assertEqual(len(indices), 16)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 63)
        self.assertTrue(np.all(np.diff(indices) > 0))

    def test_select_csi_dimensions_keeps_expected_values(self) -> None:
        csi = np.arange(2 * 8 * 8).reshape(2, 8, 8).astype(np.complex64)
        selected = select_csi_dimensions(csi, np.array([1, 4]), np.array([0, 3, 7]))
        self.assertEqual(selected.shape, (2, 2, 3))
        np.testing.assert_array_equal(selected[:, 0], csi[:, 1, [0, 3, 7]])


if __name__ == "__main__":
    unittest.main()
