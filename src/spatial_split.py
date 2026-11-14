"""Leakage-resistant spatial splitting utilities."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

SPLIT_CODES = {"guard": -1, "train": 0, "validation": 1, "test": 2}


def split_by_coordinate_columns(
    positions: np.ndarray,
    *,
    axis: str,
    regions: dict[str, list[int]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Assign entire coordinate columns to spatially disjoint regions."""
    positions = np.asarray(positions)
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError(f"Positions must have shape [users, >=2], got {positions.shape}")
    if axis not in {"x", "y"}:
        raise ValueError("Split axis must be 'x' or 'y'")

    axis_index = 0 if axis == "x" else 1
    coordinates = positions[:, axis_index]
    if not np.isfinite(coordinates).all():
        raise ValueError(f"Non-finite {axis} coordinates cannot be spatially split")
    unique_columns = np.unique(coordinates)

    required_regions = {"train", "validation", "test"}
    if set(regions) != required_regions:
        raise ValueError(f"Regions must be exactly {sorted(required_regions)}")

    codes = np.full(len(positions), SPLIT_CODES["guard"], dtype=np.int8)
    claimed_columns: set[int] = set()
    report_regions: dict[str, Any] = {}

    for name in ("validation", "train", "test"):
        start, stop = (int(value) for value in regions[name])
        if not (0 <= start < stop <= len(unique_columns)):
            raise ValueError(
                f"Invalid {name} column range [{start}, {stop}) for "
                f"{len(unique_columns)} unique {axis} columns"
            )
        column_ids = set(range(start, stop))
        if claimed_columns.intersection(column_ids):
            raise ValueError(f"Spatial column ranges overlap at region '{name}'")
        claimed_columns.update(column_ids)

        selected_values = unique_columns[start:stop]
        mask = np.isin(coordinates, selected_values)
        codes[mask] = SPLIT_CODES[name]
        selected_positions = positions[mask, :2]
        report_regions[name] = {
            "code": SPLIT_CODES[name],
            "num_users": int(mask.sum()),
            "num_columns": int(stop - start),
            "column_range": [start, stop],
            "xy_min": selected_positions.min(axis=0).astype(float).tolist(),
            "xy_max": selected_positions.max(axis=0).astype(float).tolist(),
        }

    guard_mask = codes == SPLIT_CODES["guard"]
    report_regions["guard"] = {
        "code": SPLIT_CODES["guard"],
        "num_users": int(guard_mask.sum()),
        "num_columns": int(len(unique_columns) - len(claimed_columns)),
    }

    assigned_names = ("validation", "train", "test")
    separations: dict[str, float] = {}
    for first, second in combinations(assigned_names, 2):
        first_values = np.unique(coordinates[codes == SPLIT_CODES[first]])
        second_values = np.unique(coordinates[codes == SPLIT_CODES[second]])
        separation = np.min(np.abs(first_values[:, None] - second_values[None, :]))
        separations[f"{first}_to_{second}"] = float(separation)

    retained = int((codes != SPLIT_CODES["guard"]).sum())
    report = {
        "axis": axis,
        "num_unique_columns": int(len(unique_columns)),
        "coordinate_min": float(unique_columns.min()),
        "coordinate_max": float(unique_columns.max()),
        "regions": report_regions,
        "retained_users": retained,
        "excluded_guard_users": int(guard_mask.sum()),
        "retained_fractions": {
            name: float(report_regions[name]["num_users"] / retained)
            for name in assigned_names
        },
        "axis_separation_m": separations,
    }
    return codes, report
