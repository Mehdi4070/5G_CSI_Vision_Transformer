#!/usr/bin/env python3
"""Compare ViT localization versus antenna and subcarrier counts."""

from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


BASELINE = PROJECT_ROOT / "outputs" / "vit_baseline"
STUDIES = {
    "antennas": {
        16: PROJECT_ROOT / "outputs" / "system_parameters" / "vit_antennas16",
        32: PROJECT_ROOT / "outputs" / "system_parameters" / "vit_antennas32",
        64: BASELINE,
    },
    "subcarriers": {
        16: PROJECT_ROOT / "outputs" / "system_parameters" / "vit_subcarriers16",
        32: PROJECT_ROOT / "outputs" / "system_parameters" / "vit_subcarriers32",
        64: BASELINE,
    },
}


def load_run(directory: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    with (directory / "metrics.json").open(encoding="utf-8") as stream:
        metrics = json.load(stream)
    with np.load(directory / "test_predictions.npz", allow_pickle=False) as data:
        predictions = {name: np.asarray(data[name]) for name in data.files}
    return metrics, predictions


def validate_same_test_users(
    runs: dict[int, tuple[dict[str, object], dict[str, np.ndarray]]]
) -> None:
    reference = runs[64][1]
    for count, (_, predictions) in runs.items():
        if not np.array_equal(predictions["indices"], reference["indices"]):
            raise ValueError(f"The {count}-element run uses different test indices")
        np.testing.assert_allclose(
            predictions["targets_xy"], reference["targets_xy"], atol=1e-6
        )


def save_cdf(
    name: str,
    runs: dict[int, tuple[dict[str, object], dict[str, np.ndarray]]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for count in sorted(runs):
        errors = np.sort(runs[count][1]["errors_m"])
        cdf = np.arange(1, len(errors) + 1) / len(errors)
        ax.plot(errors, cdf, linewidth=2, label=str(count))
    label = "BS antennas" if name == "antennas" else "OFDM subcarriers"
    ax.set(
        title=f"ViT test-error CDF versus {label.lower()}",
        xlabel="Localization error [m]",
        ylabel="CDF",
    )
    ax.grid(alpha=0.3)
    ax.legend(title=label)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    output_directory = PROJECT_ROOT / "outputs" / "system_parameters" / "comparison"
    output_directory.mkdir(parents=True, exist_ok=True)
    loaded = {
        study: {count: load_run(directory) for count, directory in directories.items()}
        for study, directories in STUDIES.items()
    }
    for runs in loaded.values():
        validate_same_test_users(runs)

    summary: dict[str, object] = {
        "controlled_variables": {
            "same_10000_user_positions": True,
            "same_spatial_train_validation_test_split": True,
            "same_vit_hyperparameters_except_input_size": True,
            "same_seed_and_training_configuration": True,
            "same_50_mhz_bandwidth": True,
            "num_test_users": int(len(loaded["antennas"][64][1]["indices"])),
            "antenna_geometries": {"16": "4x4", "32": "4x8", "64": "8x8"},
            "subcarrier_selection": "uniformly distributed over the original 50 MHz band",
        },
        "studies": {},
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for axis, (study, runs) in zip(axes, loaded.items(), strict=True):
        study_results: dict[str, object] = {}
        for count in sorted(runs):
            metrics = runs[count][0]
            test_metrics = metrics["vit"]["test"]["metrics"]
            study_results[str(count)] = {
                "test": test_metrics,
                "validation": metrics["vit"]["validation"]["metrics"],
                "best_epoch": metrics["best_epoch"],
                "elapsed_seconds": metrics["elapsed_seconds"],
                "parameter_count": metrics["parameter_count"],
            }
        baseline_mean = float(study_results["64"]["test"]["mean_error_m"])
        for values in study_results.values():
            current_mean = float(values["test"]["mean_error_m"])
            values["mean_error_change_vs_64_percent"] = (
                100.0 * (current_mean - baseline_mean) / baseline_mean
            )
        summary["studies"][study] = study_results

        counts = np.asarray(sorted(runs))
        for key, label, marker in (
            ("mean_error_m", "Mean", "o"),
            ("median_error_m", "Median", "s"),
            ("p90_error_m", "P90", "^"),
        ):
            values = [study_results[str(count)]["test"][key] for count in counts]
            axis.plot(counts, values, marker=marker, linewidth=2, label=label)
        parameter_label = "Number of BS antennas" if study == "antennas" else "Number of OFDM subcarriers"
        axis.set(
            title=f"Influence of {'BS antennas' if study == 'antennas' else 'OFDM subcarriers'}",
            xlabel=parameter_label,
            ylabel="Test localization error [m]",
            xticks=counts,
        )
        axis.grid(alpha=0.3)
        axis.legend()

        save_cdf(
            study,
            runs,
            output_directory / f"{study}_test_cdf.png",
        )

    fig.savefig(output_directory / "system_parameter_metrics.png", dpi=180)
    plt.close(fig)
    with (output_directory / "system_parameter_comparison.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
