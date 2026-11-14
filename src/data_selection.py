"""Pure spatial-selection helpers for DeepMIMO receiver grids."""

from __future__ import annotations

import numpy as np


def select_rectangle_user_ids(
    positions: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> np.ndarray:
    """Return grid indices inside half-open XY bounds.

    Half-open bounds make neighboring regions non-overlapping and avoid
    double-counting their shared border during later spatial splitting.
    """
    positions = np.asarray(positions)
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError(f"Positions must have shape [users, >=2], got {positions.shape}")
    if not (x_min < x_max and y_min < y_max):
        raise ValueError("Rectangle bounds must satisfy x_min < x_max and y_min < y_max")

    xy = positions[:, :2]
    finite = np.isfinite(xy).all(axis=1)
    mask = (
        finite
        & (xy[:, 0] >= x_min)
        & (xy[:, 0] < x_max)
        & (xy[:, 1] >= y_min)
        & (xy[:, 1] < y_max)
    )
    user_ids = np.flatnonzero(mask).astype(np.int64, copy=False)
    if not len(user_ids):
        raise ValueError(
            "The requested rectangle contains no users: "
            f"x=[{x_min}, {x_max}), y=[{y_min}, {y_max})"
        )
    return user_ids

