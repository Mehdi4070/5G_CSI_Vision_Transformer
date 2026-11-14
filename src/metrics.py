"""Localization metrics shared by naïve, CNN, and future ViT models."""

from __future__ import annotations

from typing import Any

import numpy as np


def localization_errors(predictions_xy: np.ndarray, targets_xy: np.ndarray) -> np.ndarray:
    predictions = np.asarray(predictions_xy, dtype=np.float64)
    targets = np.asarray(targets_xy, dtype=np.float64)
    if predictions.shape != targets.shape or predictions.ndim != 2 or predictions.shape[1] != 2:
        raise ValueError(
            f"Predictions and targets must both have shape [users, 2], got "
            f"{predictions.shape} and {targets.shape}"
        )
    return np.linalg.norm(predictions - targets, axis=1)


def localization_metrics(predictions_xy: np.ndarray, targets_xy: np.ndarray) -> dict[str, Any]:
    errors = localization_errors(predictions_xy, targets_xy)
    return {
        "num_users": int(len(errors)),
        "mean_error_m": float(errors.mean()),
        "median_error_m": float(np.median(errors)),
        "rmse_m": float(np.sqrt(np.mean(np.square(errors)))),
        "p90_error_m": float(np.percentile(errors, 90)),
        "p95_error_m": float(np.percentile(errors, 95)),
        "min_error_m": float(errors.min()),
        "max_error_m": float(errors.max()),
    }


def mean_position_baseline(
    train_positions_xy: np.ndarray, evaluation_positions_xy: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    mean_position = np.asarray(train_positions_xy, dtype=np.float64).mean(axis=0)
    predictions = np.repeat(mean_position[None, :], len(evaluation_positions_xy), axis=0)
    return predictions, localization_metrics(predictions, evaluation_positions_xy)

