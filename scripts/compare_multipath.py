#!/usr/bin/env python3
"""Compare ViT localization results for different retained path counts."""

from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


RUN_DIRECTORIES = {
    1: PROJECT_ROOT / "outputs" / "multipath" / "vit_paths1",
    3: PROJECT_ROOT / "outputs" / "multipath" / "vit_paths3",
    5: PROJECT_ROOT / "outputs" / "multipath" / "vit_paths5",
    10: PROJECT_ROOT / "outputs" / "vit_baseline",
}


def load_predictions(directory: Path) -> dict[str, np.ndarray]:
    with np.load(directory / "test_predictions.npz", allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def main() -> None:
    output_directory = PROJECT_ROOT / "outputs" / "multipath" / "comparison"
    output_directory.mkdir(parents=True, exist_ok=True)
    predictions = {
        paths: load_predictions(directory)
        for paths, directory in RUN_DIRECTORIES.items()
    }
    reference = predictions[10]
    for paths, values in predictions.items():
        if not np.array_equal(values["indices"], reference["indices"]):
            raise ValueError(f"{paths}-path run does not use the same test indices")
        np.testing.assert_allclose(values["targets_xy"], reference["targets_xy"], atol=1e-6)

    runs: dict[int, dict[str, object]] = {}
    for paths, directory in RUN_DIRECTORIES.items():
        with (directory / "metrics.json").open(encoding="utf-8") as stream:
            run = json.load(stream)
        runs[paths] = {
            "test": run["vit"]["test"]["metrics"],
            "validation": run["vit"]["validation"]["metrics"],
            "best_epoch": run["best_epoch"],
            "elapsed_seconds": run["elapsed_seconds"],
            "parameter_count": run["parameter_count"],
        }

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for paths, values in predictions.items():
        errors = np.sort(values["errors_m"])
        cdf = np.arange(1, len(errors) + 1) / len(errors)
        ax.plot(errors, cdf, label=f"{paths} path{'s' if paths > 1 else ''}")
    ax.set(
        title="ViT test-error CDF versus multipath richness",
        xlabel="Localization error [m]",
        ylabel="CDF",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(output_directory / "multipath_test_cdf.png", dpi=180)
    plt.close(fig)

    path_counts = np.asarray(sorted(runs))
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for key, label, marker in (
        ("mean_error_m", "Mean", "o"),
        ("median_error_m", "Median", "s"),
        ("p90_error_m", "P90", "^"),
    ):
        values = [float(runs[int(paths)]["test"][key]) for paths in path_counts]
        ax.plot(path_counts, values, marker=marker, linewidth=2, label=label)
    ax.set(
        title="Localization error versus retained path count",
        xlabel="Maximum retained propagation paths",
        ylabel="Test error [m]",
        xticks=path_counts,
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(output_directory / "multipath_metrics.png", dpi=180)
    plt.close(fig)

    one_path_mean = float(runs[1]["test"]["mean_error_m"])
    summary = {
        "controlled_variables": {
            "same_user_positions": True,
            "same_spatial_splits": True,
            "same_vit_architecture": True,
            "same_seed_and_training_configuration": True,
            "num_test_users": int(len(reference["indices"])),
        },
        "results": {str(paths): values for paths, values in runs.items()},
        "best_test_mean_paths": int(
            min(runs, key=lambda paths: float(runs[paths]["test"]["mean_error_m"]))
        ),
        "mean_error_change_vs_1_path_percent": {
            str(paths): 100.0
            * (float(runs[paths]["test"]["mean_error_m"]) - one_path_mean)
            / one_path_mean
            for paths in runs
        },
    }
    with (output_directory / "multipath_comparison.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
