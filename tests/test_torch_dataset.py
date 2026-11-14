import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.torch_dataset import CSILocalizationDataset


class TorchDatasetTests(unittest.TestCase):
    def test_loads_only_requested_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.save(path / "features.npy", np.zeros((5, 3, 4, 4), dtype=np.float32))
            np.save(path / "targets_normalized.npy", np.arange(10, dtype=np.float32).reshape(5, 2))
            np.savez(
                path / "split_indices.npz",
                train=np.array([0, 2, 4]),
                validation=np.array([1]),
                test=np.array([3]),
                guard=np.array([], dtype=np.int64),
            )

            dataset = CSILocalizationDataset(path, "train")
            features, target, global_index = dataset[1]

            self.assertEqual(len(dataset), 3)
            self.assertEqual(tuple(features.shape), (3, 4, 4))
            self.assertEqual(tuple(target.shape), (2,))
            self.assertEqual(global_index, 2)


if __name__ == "__main__":
    unittest.main()

