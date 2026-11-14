#!/usr/bin/env python3
"""Generate and validate a small raw CSI dataset from DeepMIMO v4."""

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

import deepmimo as dm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))
from src.data_validation import validate_raw_dataset  # noqa: E402
from src.data_selection import select_rectangle_user_ids  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Step-1 DeepMIMO CSI sample and validation artifacts."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "o1_28_small.yaml",
        help="YAML experiment configuration.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the configured DeepMIMO scenario when it is absent.",
    )
    return parser.parse_args()


def project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required_sections = {"scenario", "selection", "channel", "output"}
    missing = required_sections.difference(config or {})
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    return config


def ensure_scenario(config: dict[str, Any], *, download: bool) -> str:
    scenario_name = str(config["scenario"]["name"]).lower()
    scenario_directory = project_path(config["scenario"]["directory"])
    scenario_directory.mkdir(parents=True, exist_ok=True)
    dm.config.set("scenarios_folder", str(scenario_directory))

    scenario_path = Path(dm.get_scenario_folder(scenario_name))
    if not scenario_path.exists() and download:
        downloaded = dm.download(scenario_name)
        if downloaded is None and not scenario_path.exists():
            raise RuntimeError(f"DeepMIMO could not download scenario '{scenario_name}'")

    if not scenario_path.exists():
        command = ".venv/bin/python scripts/prepare_deepmimo.py --download"
        raise FileNotFoundError(
            f"Scenario '{scenario_name}' is not present at {scenario_path}.\n"
            f"Run with download enabled:\n  {command}"
        )
    return scenario_name


def selected_index_user_ids(selection: dict[str, Any]) -> np.ndarray:
    first = int(selection["first_user"])
    count = int(selection["num_users"])
    stride = int(selection["user_stride"])
    if first < 0 or count <= 0 or stride <= 0:
        raise ValueError("first_user must be >= 0; num_users and user_stride must be > 0")
    return first + np.arange(count, dtype=np.int64) * stride


def selected_user_ids(scenario_name: str, selection: dict[str, Any]) -> np.ndarray:
    mode = str(selection.get("mode", "indices")).lower()
    if mode == "indices":
        return selected_index_user_ids(selection)
    if mode != "rectangle":
        raise ValueError(f"Unknown selection mode '{mode}'; expected 'indices' or 'rectangle'")

    print("Loading receiver coordinates to resolve the 2D rectangle...")
    positions_dataset = dm.load(
        scenario_name,
        tx_sets={int(selection["tx_set"]): [int(selection["tx_point"])]},
        rx_sets={int(selection["rx_set"]): "all"},
        matrices=["rx_pos", "tx_pos"],
    )
    user_ids = select_rectangle_user_ids(
        positions_dataset.rx_pos,
        x_min=float(selection["x_min"]),
        x_max=float(selection["x_max"]),
        y_min=float(selection["y_min"]),
        y_max=float(selection["y_max"]),
    )
    expected = selection.get("expected_num_users")
    if expected is not None and len(user_ids) != int(expected):
        raise ValueError(
            f"Rectangle selected {len(user_ids)} users, but configuration expects {expected}"
        )
    return user_ids


def generate_dataset(
    scenario_name: str, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selection = config["selection"]
    channel = config["channel"]
    user_ids = selected_user_ids(scenario_name, selection)

    print(f"Loading {len(user_ids)} users from scenario '{scenario_name}'...")
    dataset = dm.load(
        scenario_name,
        tx_sets={int(selection["tx_set"]): [int(selection["tx_point"])]},
        rx_sets={int(selection["rx_set"]): user_ids},
        max_paths=int(channel["num_paths"]),
    )

    parameters = dm.ChannelParameters()
    parameters.freq_domain = True
    parameters.num_paths = int(channel["num_paths"])
    parameters.bs_antenna.shape = np.asarray(channel["bs_antenna_shape"], dtype=int)
    parameters.ue_antenna.shape = np.asarray(channel["ue_antenna_shape"], dtype=int)
    spacing = float(channel["antenna_spacing_wavelengths"])
    parameters.bs_antenna.spacing = spacing
    parameters.ue_antenna.spacing = spacing
    parameters.ofdm.bandwidth = float(channel["bandwidth_hz"])
    parameters.ofdm.subcarriers = int(channel["num_subcarriers"])
    parameters.ofdm.selected_subcarriers = np.arange(
        int(channel["num_subcarriers"]), dtype=int
    )

    print("Computing frequency-domain CSI...")
    channels = np.asarray(dataset.compute_channels(parameters))
    if channels.ndim != 4:
        raise ValueError(
            "DeepMIMO channels must have shape [users, UE antennas, BS antennas, subcarriers], "
            f"got {channels.shape}"
        )
    if channels.shape[1] != 1:
        raise ValueError(
            "Step 1 expects one UE antenna so CSI can be represented as "
            f"[BS antennas, subcarriers], got {channels.shape[1]} UE antennas"
        )

    csi = channels[:, 0, :, :].astype(np.complex64, copy=False)
    positions = np.asarray(dataset.rx_pos, dtype=np.float32)
    tx_positions = np.asarray(dataset.tx_pos, dtype=np.float32)
    if len(csi) != len(user_ids):
        raise ValueError(
            f"DeepMIMO returned {len(csi)} users for {len(user_ids)} requested IDs; "
            "the CSI-to-ID association is therefore ambiguous."
        )
    return csi, positions, user_ids, tx_positions


def save_position_plot(positions: np.ndarray, tx_positions: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.scatter(positions[:, 0], positions[:, 1], s=8, alpha=0.7, label="UE positions")
    tx = np.atleast_2d(tx_positions)
    ax.scatter(tx[:, 0], tx[:, 1], marker="^", s=110, c="red", label="Base station")
    ax.set(title="DeepMIMO user region", xlabel="x [m]", ylabel="y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_csi_plots(csi: np.ndarray, valid_mask: np.ndarray, output_dir: Path) -> int:
    valid_indices = np.flatnonzero(valid_mask)
    if not len(valid_indices):
        raise ValueError("No valid CSI sample is available for visualization")
    sample_index = int(valid_indices[0])
    sample = csi[sample_index]

    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(sample), 1e-12))
    phase = np.angle(sample)
    for values, name, title, colorbar_label, cmap in (
        (magnitude_db, "csi_amplitude_db.png", "CSI amplitude", "Amplitude [dB]", "viridis"),
        (phase, "csi_phase.png", "CSI phase", "Phase [rad]", "twilight"),
    ):
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        image = ax.imshow(values, aspect="auto", origin="lower", cmap=cmap)
        ax.set(title=f"{title} — sample {sample_index}", xlabel="OFDM subcarrier", ylabel="BS antenna")
        fig.colorbar(image, ax=ax, label=colorbar_label)
        fig.savefig(output_dir / name, dpi=180)
        plt.close(fig)
    return sample_index


def enforce_validation_requirements(
    report: dict[str, Any], validation: dict[str, Any]
) -> None:
    minimum_span = validation.get("min_xy_span_m")
    if minimum_span is not None:
        actual = [
            report["position_coverage"]["x"]["span_m"],
            report["position_coverage"]["y"]["span_m"],
        ]
        if any(value < minimum for value, minimum in zip(actual, minimum_span, strict=True)):
            raise ValueError(
                f"XY coverage {actual} m is below the required minimum {minimum_span} m"
            )
    if validation.get("require_all_users_valid", False) and report["invalid_users"]:
        raise ValueError(
            f"Validation requires all users to be valid, but {report['invalid_users']} are invalid"
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    scenario_name = ensure_scenario(config, download=args.download)
    csi, positions, user_ids, tx_positions = generate_dataset(scenario_name, config)

    report, valid_mask = validate_raw_dataset(csi, positions, user_ids)
    enforce_validation_requirements(report, config.get("validation", {}))
    output_dir = project_path(config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / config["output"]["dataset_filename"]
    np.savez_compressed(
        dataset_path,
        csi=csi,
        positions=positions,
        user_ids=user_ids,
        valid_mask=valid_mask,
        tx_positions=tx_positions,
    )

    save_position_plot(positions, tx_positions, output_dir / "user_positions.png")
    report["visualized_sample_index"] = save_csi_plots(csi, valid_mask, output_dir)
    report["scenario"] = scenario_name
    report["deepmimo_version"] = dm.__version__
    report["configuration"] = config
    report["dataset_file"] = str(dataset_path.relative_to(PROJECT_ROOT))

    report_path = output_dir / config["output"]["report_filename"]
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)

    print("\nStep 1 artifacts created:")
    print(f"  Dataset: {dataset_path}")
    print(f"  Report:  {report_path}")
    print(f"  Valid users: {report['valid_users']}/{report['num_users']}")
    print(f"  CSI shape: {tuple(report['csi_shape'])}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from None
