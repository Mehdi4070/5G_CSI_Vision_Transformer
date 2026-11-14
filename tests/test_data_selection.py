import unittest

import numpy as np

from src.data_selection import select_rectangle_user_ids


class RectangleSelectionTests(unittest.TestCase):
    def test_uses_half_open_bounds(self) -> None:
        positions = np.array(
            [
                [0.0, 0.0],
                [0.5, 0.5],
                [1.0, 0.5],
                [0.5, 1.0],
                [np.nan, 0.5],
            ]
        )

        selected = select_rectangle_user_ids(
            positions, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0
        )

        np.testing.assert_array_equal(selected, [0, 1])

    def test_rejects_invalid_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "x_min < x_max"):
            select_rectangle_user_ids(
                np.zeros((2, 2)), x_min=1.0, x_max=0.0, y_min=0.0, y_max=1.0
            )

    def test_rejects_empty_rectangle(self) -> None:
        with self.assertRaisesRegex(ValueError, "contains no users"):
            select_rectangle_user_ids(
                np.zeros((2, 2)), x_min=2.0, x_max=3.0, y_min=2.0, y_max=3.0
            )


if __name__ == "__main__":
    unittest.main()

