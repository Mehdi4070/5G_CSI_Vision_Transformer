"""Controlled antenna and OFDM-subcarrier selection for CSI ablations."""

from __future__ import annotations

import numpy as np


def centered_subarray_indices(
    source_shape: tuple[int, int], target_shape: tuple[int, int]
) -> np.ndarray:
    """Return flattened indices of a centered rectangular physical subarray."""
    source = np.asarray(source_shape, dtype=np.int64)
    target = np.asarray(target_shape, dtype=np.int64)
    if source.shape != (2,) or target.shape != (2,):
        raise ValueError("Antenna shapes must contain exactly two dimensions")
    if np.any(source <= 0) or np.any(target <= 0):
        raise ValueError("Antenna dimensions must be positive")
    if np.any(target > source):
        raise ValueError(
            f"Target antenna shape {tuple(target)} exceeds source shape {tuple(source)}"
        )

    start = (source - target) // 2
    rows = np.arange(start[0], start[0] + target[0])
    columns = np.arange(start[1], start[1] + target[1])
    row_grid, column_grid = np.meshgrid(rows, columns, indexing="ij")
    return np.ravel_multi_index(
        (row_grid.ravel(), column_grid.ravel()), tuple(source)
    ).astype(np.int64)


def uniform_subcarrier_indices(total: int, selected: int) -> np.ndarray:
    """Select frequency bins approximately uniformly across the full bandwidth."""
    if total <= 0 or selected <= 0:
        raise ValueError("Subcarrier counts must be positive")
    if selected > total:
        raise ValueError(f"Cannot select {selected} subcarriers from {total}")
    if selected == total:
        return np.arange(total, dtype=np.int64)

    indices = np.rint(np.linspace(0, total - 1, selected)).astype(np.int64)
    if len(np.unique(indices)) != selected:
        raise RuntimeError("Uniform subcarrier selection produced duplicate indices")
    return indices


def select_csi_dimensions(
    csi: np.ndarray,
    antenna_indices: np.ndarray,
    subcarrier_indices: np.ndarray,
) -> np.ndarray:
    """Select both axes while preserving ``[users, antennas, subcarriers]``."""
    csi = np.asarray(csi)
    antenna_indices = np.asarray(antenna_indices, dtype=np.int64)
    subcarrier_indices = np.asarray(subcarrier_indices, dtype=np.int64)
    if csi.ndim != 3 or not np.iscomplexobj(csi):
        raise ValueError("CSI must be complex with shape [users, antennas, subcarriers]")
    if antenna_indices.ndim != 1 or subcarrier_indices.ndim != 1:
        raise ValueError("Selected antenna and subcarrier indices must be one-dimensional")
    if not len(antenna_indices) or not len(subcarrier_indices):
        raise ValueError("At least one antenna and one subcarrier must be selected")
    if (
        antenna_indices.min() < 0
        or antenna_indices.max() >= csi.shape[1]
        or subcarrier_indices.min() < 0
        or subcarrier_indices.max() >= csi.shape[2]
    ):
        raise ValueError("Selected CSI dimension index is outside the source tensor")
    return csi[:, antenna_indices][:, :, subcarrier_indices]
