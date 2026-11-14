"""Complex AWGN utilities for CSI robustness experiments."""

from __future__ import annotations

import numpy as np


def normalized_complex_noise(
    shape: tuple[int, ...], rng: np.random.Generator
) -> np.ndarray:
    """Create per-sample unit-power circular complex Gaussian noise."""
    if len(shape) != 3 or any(size <= 0 for size in shape):
        raise ValueError("Noise shape must be [users, antennas, subcarriers]")
    real = rng.standard_normal(shape, dtype=np.float32)
    imaginary = rng.standard_normal(shape, dtype=np.float32)
    noise = (real + 1j * imaginary).astype(np.complex64)
    power = np.mean(np.abs(noise) ** 2, axis=(1, 2), keepdims=True)
    return (noise / np.sqrt(power)).astype(np.complex64)


def add_complex_awgn(
    csi: np.ndarray, *, snr_db: float, unit_noise: np.ndarray
) -> np.ndarray:
    """Add AWGN at an exact per-user SNR relative to mean CSI power."""
    csi = np.asarray(csi)
    unit_noise = np.asarray(unit_noise)
    if csi.ndim != 3 or not np.iscomplexobj(csi):
        raise ValueError("CSI must be complex with shape [users, antennas, subcarriers]")
    if unit_noise.shape != csi.shape or not np.iscomplexobj(unit_noise):
        raise ValueError("unit_noise must be complex and have the same shape as CSI")
    signal_power = np.mean(np.abs(csi) ** 2, axis=(1, 2), keepdims=True)
    snr_linear = np.float32(10.0 ** (float(snr_db) / 10.0))
    noise_scale = np.sqrt(signal_power / snr_linear)
    return (csi + noise_scale * unit_noise).astype(np.complex64)


def measured_snr_db(clean_csi: np.ndarray, noisy_csi: np.ndarray) -> np.ndarray:
    """Measure per-user SNR from clean and noisy CSI arrays."""
    clean = np.asarray(clean_csi)
    noisy = np.asarray(noisy_csi)
    if clean.shape != noisy.shape:
        raise ValueError("Clean and noisy CSI must have the same shape")
    signal_power = np.mean(np.abs(clean) ** 2, axis=(1, 2))
    noise_power = np.mean(np.abs(noisy - clean) ** 2, axis=(1, 2))
    return 10.0 * np.log10(signal_power / noise_power)
