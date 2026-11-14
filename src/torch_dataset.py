"""Memory-mapped PyTorch dataset for CSI localization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CSILocalizationDataset(Dataset):
    """Expose one spatial split without loading the feature tensor into RAM."""

    def __init__(
        self,
        data_directory: str | Path,
        split: str,
        *,
        limit: int | None = None,
    ) -> None:
        self.data_directory = Path(data_directory)
        self.features = np.load(self.data_directory / "features.npy", mmap_mode="r")
        self.targets = np.load(self.data_directory / "targets_normalized.npy", mmap_mode="r")
        with np.load(self.data_directory / "split_indices.npz", allow_pickle=False) as split_file:
            if split not in split_file.files:
                raise ValueError(f"Unknown split '{split}'; available: {split_file.files}")
            indices = np.asarray(split_file[split], dtype=np.int64)
        if limit is not None:
            if limit <= 0:
                raise ValueError("Dataset limit must be positive")
            indices = indices[:limit]
        self.indices = indices
        self.split = split

        if len(self.features) != len(self.targets):
            raise ValueError("Features and targets contain different numbers of users")
        if self.features.ndim != 4 or self.features.shape[1] != 3:
            raise ValueError(f"Expected features [users, 3, H, W], got {self.features.shape}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        global_index = int(self.indices[item])
        # Each sample is copied from the read-only memmap into writable batch memory.
        features = torch.from_numpy(np.array(self.features[global_index], dtype=np.float32, copy=True))
        target = torch.from_numpy(np.array(self.targets[global_index], dtype=np.float32, copy=True))
        return features, target, global_index

