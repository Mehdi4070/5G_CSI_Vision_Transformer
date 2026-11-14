"""Integrity checks for raw CSI arrays and their position labels."""

from __future__ import annotations

from typing import Any

import numpy as np


def validate_raw_dataset(
    csi: np.ndarray,
    positions: np.ndarray,
    user_ids: np.ndarray,
    *,
    zero_threshold: float = 1e-12,
) -> tuple[dict[str, Any], np.ndarray]:
    """Validate one-to-one CSI/position associations.

    The function does not silently remove invalid users. It returns a mask so
    later preprocessing can make that decision explicitly.
    """
    csi = np.asarray(csi)
    positions = np.asarray(positions)
    user_ids = np.asarray(user_ids)

    if csi.ndim != 3:
        raise ValueError(f"CSI must have shape [users, antennas, subcarriers], got {csi.shape}")
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError(f"Positions must have shape [users, >=2], got {positions.shape}")
    if user_ids.ndim != 1:
        raise ValueError(f"User IDs must be one-dimensional, got {user_ids.shape}")
    if not (len(csi) == len(positions) == len(user_ids)):
        raise ValueError(
            "CSI, positions and user IDs must contain the same number of users: "
            f"{len(csi)}, {len(positions)}, {len(user_ids)}"
        )
    if not np.iscomplexobj(csi):
        raise ValueError("Raw CSI must remain complex-valued during Step 1")

    finite_csi = np.isfinite(csi.real) & np.isfinite(csi.imag)
    finite_csi_per_user = finite_csi.all(axis=(1, 2))
    finite_positions_per_user = np.isfinite(positions).all(axis=1)

    magnitudes = np.abs(csi)
    zero_csi_per_user = np.max(magnitudes, axis=(1, 2)) <= zero_threshold
    valid_mask = finite_csi_per_user & finite_positions_per_user & ~zero_csi_per_user

    finite_xy = positions[finite_positions_per_user, :2]
    unique_xy = np.unique(finite_xy, axis=0) if len(finite_xy) else finite_xy
    duplicate_xy_count = int(len(finite_xy) - len(unique_xy))

    axis_names = ("x", "y", "z")
    position_coverage: dict[str, Any] = {}
    for axis, name in enumerate(axis_names[: positions.shape[1]]):
        values = positions[finite_positions_per_user, axis]
        position_coverage[name] = {
            "min": float(values.min()) if len(values) else None,
            "max": float(values.max()) if len(values) else None,
            "span_m": float(np.ptp(values)) if len(values) else None,
            "unique_values": int(len(np.unique(values))),
        }

    finite_magnitudes = magnitudes[finite_csi]
    if finite_magnitudes.size:
        magnitude_stats = {
            "min": float(finite_magnitudes.min()),
            "median": float(np.median(finite_magnitudes)),
            "mean": float(finite_magnitudes.mean()),
            "max": float(finite_magnitudes.max()),
        }
    else:
        magnitude_stats = {key: None for key in ("min", "median", "mean", "max")}

    report: dict[str, Any] = {
        "num_users": int(len(csi)),
        "csi_shape": list(csi.shape),
        "positions_shape": list(positions.shape),
        "csi_dtype": str(csi.dtype),
        "positions_dtype": str(positions.dtype),
        "is_complex": True,
        "finite_csi_users": int(finite_csi_per_user.sum()),
        "finite_position_users": int(finite_positions_per_user.sum()),
        "zero_csi_users": int(zero_csi_per_user.sum()),
        "duplicate_xy_positions": duplicate_xy_count,
        "valid_users": int(valid_mask.sum()),
        "invalid_users": int((~valid_mask).sum()),
        "position_coverage": position_coverage,
        "csi_magnitude": magnitude_stats,
    }
    return report, valid_mask
