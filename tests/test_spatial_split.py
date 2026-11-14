import unittest

import numpy as np

from src.spatial_split import SPLIT_CODES, split_by_coordinate_columns


class SpatialSplitTests(unittest.TestCase):
    def test_assigns_whole_columns_and_guard_zones(self) -> None:
        x, y = np.meshgrid(np.arange(10, dtype=float), np.arange(3, dtype=float))
        positions = np.column_stack((x.ravel(), y.ravel()))

        codes, report = split_by_coordinate_columns(
            positions,
            axis="x",
            regions={"validation": [0, 2], "train": [3, 7], "test": [8, 10]},
        )

        self.assertEqual(np.sum(codes == SPLIT_CODES["validation"]), 6)
        self.assertEqual(np.sum(codes == SPLIT_CODES["train"]), 12)
        self.assertEqual(np.sum(codes == SPLIT_CODES["test"]), 6)
        self.assertEqual(np.sum(codes == SPLIT_CODES["guard"]), 6)
        self.assertEqual(report["axis_separation_m"]["validation_to_train"], 2.0)
        self.assertEqual(report["axis_separation_m"]["train_to_test"], 2.0)

    def test_rejects_overlapping_regions(self) -> None:
        positions = np.column_stack((np.arange(6, dtype=float), np.zeros(6)))
        with self.assertRaisesRegex(ValueError, "overlap"):
            split_by_coordinate_columns(
                positions,
                axis="x",
                regions={"validation": [0, 2], "train": [1, 4], "test": [4, 6]},
            )


if __name__ == "__main__":
    unittest.main()

