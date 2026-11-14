import unittest

import numpy as np

from src.data_validation import validate_raw_dataset


class ValidateRawDatasetTests(unittest.TestCase):
    def test_reports_valid_complex_dataset(self) -> None:
        csi = np.ones((3, 4, 5), dtype=np.complex64) * (1 + 2j)
        positions = np.array([[0, 0, 1.5], [1, 0, 1.5], [2, 0, 1.5]], dtype=np.float32)
        user_ids = np.array([10, 11, 12])

        report, mask = validate_raw_dataset(csi, positions, user_ids)

        self.assertEqual(report["valid_users"], 3)
        self.assertEqual(report["csi_shape"], [3, 4, 5])
        self.assertEqual(report["position_coverage"]["x"]["span_m"], 2.0)
        self.assertEqual(report["position_coverage"]["y"]["unique_values"], 1)
        np.testing.assert_array_equal(mask, [True, True, True])

    def test_marks_zero_and_nonfinite_users_invalid(self) -> None:
        csi = np.ones((3, 2, 2), dtype=np.complex64)
        csi[1] = 0
        csi[2, 0, 0] = np.nan + 0j
        positions = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float32)

        report, mask = validate_raw_dataset(csi, positions, np.arange(3))

        self.assertEqual(report["zero_csi_users"], 1)
        self.assertEqual(report["invalid_users"], 2)
        np.testing.assert_array_equal(mask, [True, False, False])

    def test_rejects_real_valued_csi(self) -> None:
        with self.assertRaisesRegex(ValueError, "complex-valued"):
            validate_raw_dataset(np.ones((2, 3, 4)), np.ones((2, 2)), np.arange(2))


if __name__ == "__main__":
    unittest.main()
