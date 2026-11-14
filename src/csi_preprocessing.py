"""CSI feature transforms and train-only standardization."""

from __future__ import annotations

from typing import Any

import numpy as np

FEATURE_NAMES = ("log_amplitude_db", "cos_phase", "sin_phase")


def csi_to_features(csi: np.ndarray, *, amplitude_floor: float) -> np.ndarray:
    """Convert complex CSI to amplitude and continuous phase channels."""
    csi = np.asarray(csi)
    if csi.ndim != 3 or not np.iscomplexobj(csi):
        raise ValueError("CSI must be complex with shape [users, antennas, subcarriers]")
    if amplitude_floor <= 0:
        raise ValueError("amplitude_floor must be positive")

    features = np.empty((len(csi), 3, csi.shape[1], csi.shape[2]), dtype=np.float32)
    amplitude = np.maximum(np.abs(csi), amplitude_floor)
    phase = np.angle(csi)
    features[:, 0] = 20.0 * np.log10(amplitude)
    features[:, 1] = np.cos(phase)
    features[:, 2] = np.sin(phase)
    return features


def fit_feature_standardizer(
    csi: np.ndarray,
    train_indices: np.ndarray,
    *,
    amplitude_floor: float,
    chunk_size: int,
) -> dict[str, Any]:
    """Fit per-channel mean/std using training CSI and no other split."""
    train_indices = np.asarray(train_indices, dtype=np.int64)
    if not len(train_indices):
        raise ValueError("Cannot fit feature statistics without training users")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sum_sq = np.zeros(3, dtype=np.float64)
    count = 0
    for start in range(0, len(train_indices), chunk_size):
        indices = train_indices[start : start + chunk_size]
        features = csi_to_features(csi[indices], amplitude_floor=amplitude_floor)
        channel_sum += features.sum(axis=(0, 2, 3), dtype=np.float64)
        channel_sum_sq += np.square(features, dtype=np.float64).sum(axis=(0, 2, 3))
        count += int(features.shape[0] * features.shape[2] * features.shape[3])

    mean = channel_sum / count
    variance = np.maximum(channel_sum_sq / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    if np.any(std < 1e-8):
        raise ValueError(f"Near-zero feature standard deviation: {std.tolist()}")
    return {
        "feature_names": list(FEATURE_NAMES),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "fitted_values_per_channel": count,
        "fitted_users": int(len(train_indices)),
    }


def standardize_features(
    features: np.ndarray, *, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Apply per-channel standardization while retaining float32 storage."""
    features = np.asarray(features, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32).reshape(1, -1, 1, 1)
    std = np.asarray(std, dtype=np.float32).reshape(1, -1, 1, 1)
    if features.ndim != 4 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"Expected feature shape [users, 3, antennas, subcarriers], got {features.shape}")
    return ((features - mean) / std).astype(np.float32, copy=False)


def fit_target_standardizer(
    positions_xy: np.ndarray, train_indices: np.ndarray
) -> dict[str, Any]:
    """Fit coordinate normalization using training positions only."""
    train_targets = np.asarray(positions_xy, dtype=np.float64)[train_indices]
    mean = train_targets.mean(axis=0)
    std = train_targets.std(axis=0)
    if np.any(std < 1e-8):
        raise ValueError(f"Near-zero target standard deviation: {std.tolist()}")
    return {"mean": mean.tolist(), "std": std.tolist(), "fitted_users": int(len(train_targets))}

