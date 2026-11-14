#!/usr/bin/env python3
"""Create leakage-free model inputs from the validated raw CSI dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))
from src.csi_preprocessing import (  # noqa: E402
    csi_to_features,
    fit_feature_standardizer,
    fit_target_standardizer,
    standardize_features,
)
from src.csi_dimensions import (  # noqa: E402
    centered_subarray_indices,
    select_csi_dimensions,
    uniform_subcarrier_indices,
)
from src.spatial_split import SPLIT_CODES, split_by_coordinate_columns  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess CSI for CNN/ViT training.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "preprocess_o1_28_2d.yaml",
    )
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required = {"input", "split", "features", "output"}
    missing = required.difference(config or {})
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    return config


def stream_channel_stats(
    array: np.ndarray, indices: np.ndarray, *, chunk_size: int
) -> dict[str, list[float]]:
    channel_sum = np.zeros(array.shape[1], dtype=np.float64)
    channel_sum_sq = np.zeros(array.shape[1], dtype=np.float64)
    count = 0
    for start in range(0, len(indices), chunk_size):
        values = np.asarray(array[indices[start : start + chunk_size]], dtype=np.float32)
        channel_sum += values.sum(axis=(0, 2, 3), dtype=np.float64)
        channel_sum_sq += np.square(values, dtype=np.float64).sum(axis=(0, 2, 3))
        count += int(values.shape[0] * values.shape[2] * values.shape[3])
    mean = channel_sum / count
    std = np.sqrt(np.maximum(channel_sum_sq / count - np.square(mean), 0.0))
    return {"mean": mean.tolist(), "std": std.tolist()}


def save_split_plot(
    positions: np.ndarray, tx_positions: np.ndarray, codes: np.ndarray, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    styles = {
        "guard": ("lightgray", 8, 0.8),
        "train": ("tab:blue", 9, 0.8),
        "validation": ("tab:orange", 9, 0.8),
        "test": ("tab:green", 9, 0.8),
    }
    for name in ("guard", "validation", "train", "test"):
        mask = codes == SPLIT_CODES[name]
        color, size, alpha = styles[name]
        ax.scatter(positions[mask, 0], positions[mask, 1], s=size, c=color, alpha=alpha, label=name)
    tx = np.atleast_2d(tx_positions)
    ax.scatter(tx[:, 0], tx[:, 1], marker="^", s=120, c="red", label="base station")
    ax.set(title="Leakage-resistant spatial split", xlabel="x [m]", ylabel="y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_feature_plot(features: np.ndarray, sample_index: int, path: Path) -> None:
    names = ("Normalized log-amplitude", "Normalized cos(phase)", "Normalized sin(phase)")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for channel, (axis, name) in enumerate(zip(axes, names, strict=True)):
        image = axis.imshow(features[sample_index, channel], aspect="auto", origin="lower")
        axis.set(title=name, xlabel="OFDM subcarrier", ylabel="BS antenna")
        fig.colorbar(image, ax=axis, shrink=0.8)
    fig.suptitle(f"Model-ready CSI features — sample {sample_index}")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    input_path = project_path(config["input"]["dataset"])
    if not input_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {input_path}")

    print(f"Loading raw dataset: {input_path}")
    with np.load(input_path, allow_pickle=False) as raw:
        csi = raw["csi"]
        positions = raw["positions"]
        user_ids = raw["user_ids"]
        valid_mask = raw["valid_mask"]
        tx_positions = raw["tx_positions"]

    source_csi_shape = list(csi.shape)
    system_selection = config.get("system_selection", {})
    source_antenna_shape = tuple(
        int(value)
        for value in system_selection.get("source_antenna_shape", [csi.shape[1], 1])
    )
    target_antenna_shape = tuple(
        int(value)
        for value in system_selection.get("target_antenna_shape", source_antenna_shape)
    )
    if int(np.prod(source_antenna_shape)) != csi.shape[1]:
        raise ValueError(
            f"Source antenna shape {source_antenna_shape} does not match "
            f"the {csi.shape[1]} raw CSI antennas"
        )
    antenna_indices = centered_subarray_indices(
        source_antenna_shape, target_antenna_shape
    )
    selected_subcarriers = int(
        system_selection.get("num_subcarriers", csi.shape[2])
    )
    subcarrier_indices = uniform_subcarrier_indices(
        csi.shape[2], selected_subcarriers
    )
    csi = select_csi_dimensions(csi, antenna_indices, subcarrier_indices)
    print(
        f"Selected CSI dimensions: {csi.shape[1]} antennas "
        f"({target_antenna_shape[0]}x{target_antenna_shape[1]}) x "
        f"{csi.shape[2]} subcarriers"
    )

    if config["split"].get("require_all_raw_users_valid", False) and not valid_mask.all():
        raise ValueError(f"Raw dataset contains {int((~valid_mask).sum())} invalid users")

    split_codes, split_report = split_by_coordinate_columns(
        positions,
        axis=str(config["split"]["axis"]),
        regions=config["split"]["regions"],
    )
    split_indices = {
        name: np.flatnonzero(split_codes == code)
        for name, code in SPLIT_CODES.items()
    }
    train_indices = split_indices["train"]

    amplitude_floor = float(config["features"]["amplitude_floor"])
    chunk_size = int(config["features"]["chunk_size"])
    print("Fitting feature statistics on training users only...")
    feature_stats = fit_feature_standardizer(
        csi,
        train_indices,
        amplitude_floor=amplitude_floor,
        chunk_size=chunk_size,
    )
    target_stats = fit_target_standardizer(positions[:, :2], train_indices)

    output_dir = project_path(config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / config["output"]["features_file"]
    feature_shape = (len(csi), 3, csi.shape[1], csi.shape[2])
    model_features = np.lib.format.open_memmap(
        feature_path, mode="w+", dtype=np.float32, shape=feature_shape
    )

    print(f"Writing standardized features in chunks of {chunk_size} users...")
    mean = np.asarray(feature_stats["mean"])
    std = np.asarray(feature_stats["std"])
    for start in range(0, len(csi), chunk_size):
        stop = min(start + chunk_size, len(csi))
        features = csi_to_features(csi[start:stop], amplitude_floor=amplitude_floor)
        model_features[start:stop] = standardize_features(features, mean=mean, std=std)
    model_features.flush()

    positions_xy = positions[:, :2].astype(np.float32, copy=False)
    target_mean = np.asarray(target_stats["mean"], dtype=np.float32)
    target_std = np.asarray(target_stats["std"], dtype=np.float32)
    normalized_targets = ((positions_xy - target_mean) / target_std).astype(np.float32)

    np.save(output_dir / config["output"]["positions_file"], positions_xy)
    np.save(output_dir / config["output"]["normalized_targets_file"], normalized_targets)
    np.save(output_dir / config["output"]["user_ids_file"], user_ids)
    np.save(output_dir / config["output"]["split_codes_file"], split_codes)
    np.savez_compressed(
        output_dir / config["output"]["split_indices_file"],
        **split_indices,
    )

    train_feature_check = stream_channel_stats(
        model_features, train_indices, chunk_size=chunk_size
    )
    train_target_check = {
        "mean": normalized_targets[train_indices].mean(axis=0, dtype=np.float64).tolist(),
        "std": normalized_targets[train_indices].std(axis=0, dtype=np.float64).tolist(),
    }
    save_split_plot(positions_xy, tx_positions, split_codes, output_dir / "spatial_split.png")
    save_feature_plot(model_features, int(train_indices[0]), output_dir / "feature_example.png")

    metadata = {
        "input_dataset": str(input_path.relative_to(PROJECT_ROOT)),
        "source_csi_shape": source_csi_shape,
        "num_users": int(len(csi)),
        "feature_shape": list(feature_shape),
        "feature_dtype": "float32",
        "feature_representation": config["features"]["names"],
        "feature_standardization": feature_stats,
        "target_standardization": target_stats,
        "split": split_report,
        "post_transform_train_check": {
            "features": train_feature_check,
            "targets": train_target_check,
        },
        "files": {key: value for key, value in config["output"].items() if key != "directory"},
        "configuration": config,
        "system_selection": {
            "source_antenna_shape": list(source_antenna_shape),
            "target_antenna_shape": list(target_antenna_shape),
            "antenna_indices": antenna_indices.tolist(),
            "num_subcarriers": int(len(subcarrier_indices)),
            "subcarrier_indices": subcarrier_indices.tolist(),
            "full_bandwidth_sampling": True,
        },
    }
    metadata_path = output_dir / config["output"]["metadata_file"]
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)

    print("\nStep 2 artifacts created:")
    print(f"  Features: {feature_path}")
    print(f"  Metadata: {metadata_path}")
    for name in ("train", "validation", "test", "guard"):
        print(f"  {name}: {len(split_indices[name])} users")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from None
