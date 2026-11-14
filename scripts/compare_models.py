#!/usr/bin/env python3
"""Compare all localization models on the identical test users."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CNN, ViT, and hybrid results.")
    parser.add_argument(
        "--cnn-dir", type=Path, default=PROJECT_ROOT / "outputs" / "cnn_baseline"
    )
    parser.add_argument(
        "--vit-dir", type=Path, default=PROJECT_ROOT / "outputs" / "vit_baseline"
    )
    parser.add_argument(
        "--hybrid-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "hybrid_baseline",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "comparison"
    )
    return parser.parse_args()


def load_predictions(directory: Path) -> dict[str, np.ndarray]:
    with np.load(directory / "test_predictions.npz", allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cnn = load_predictions(args.cnn_dir)
    vit = load_predictions(args.vit_dir)
    hybrid = load_predictions(args.hybrid_dir)
    if not (
        np.array_equal(cnn["indices"], vit["indices"])
        and np.array_equal(cnn["indices"], hybrid["indices"])
    ):
        raise ValueError("Models do not use identical test indices")
    np.testing.assert_allclose(cnn["targets_xy"], vit["targets_xy"], atol=1e-6)
    np.testing.assert_allclose(cnn["targets_xy"], hybrid["targets_xy"], atol=1e-6)

    with (args.cnn_dir / "metrics.json").open(encoding="utf-8") as stream:
        cnn_run = json.load(stream)
    with (args.vit_dir / "metrics.json").open(encoding="utf-8") as stream:
        vit_run = json.load(stream)
    with (args.hybrid_dir / "metrics.json").open(encoding="utf-8") as stream:
        hybrid_run = json.load(stream)
    cnn_metrics = cnn_run["cnn"]["test"]["metrics"]
    vit_metrics = vit_run["vit"]["test"]["metrics"]
    hybrid_metrics = hybrid_run["hybrid"]["test"]["metrics"]
    naive_metrics = vit_run["naive_mean_position"]["test"]

    data_directory = PROJECT_ROOT / "data" / "processed" / "o1_28_2d"
    positions = np.load(data_directory / "positions_xy.npy", mmap_mode="r")
    with np.load(data_directory / "split_indices.npz", allow_pickle=False) as splits:
        train_mean = np.asarray(positions[splits["train"]]).mean(axis=0)
    naive_errors = np.linalg.norm(vit["targets_xy"] - train_mean, axis=1)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for errors, label in (
        (naive_errors, "Naive mean position"),
        (cnn["errors_m"], "CNN"),
        (vit["errors_m"], "ViT"),
        (hybrid["errors_m"], "CNN-Transformer"),
    ):
        ordered = np.sort(errors)
        ax.plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered), label=label)
    ax.set(
        title="Test localization-error CDF",
        xlabel="Localization error [m]",
        ylabel="CDF",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(args.output_dir / "all_models_test_cdf.png", dpi=180)
    plt.close(fig)

    summary = {
        "same_test_indices": True,
        "num_test_users": int(len(vit["indices"])),
        "naive": naive_metrics,
        "cnn": cnn_metrics,
        "vit": vit_metrics,
        "hybrid": hybrid_metrics,
        "runs": {
            "cnn": {
                "parameter_count": cnn_run["parameter_count"],
                "best_epoch": cnn_run["best_epoch"],
                "elapsed_seconds": cnn_run["elapsed_seconds"],
            },
            "vit": {
                "parameter_count": vit_run["parameter_count"],
                "best_epoch": vit_run["best_epoch"],
                "elapsed_seconds": vit_run["elapsed_seconds"],
            },
            "hybrid": {
                "parameter_count": hybrid_run["parameter_count"],
                "best_epoch": hybrid_run["best_epoch"],
                "elapsed_seconds": hybrid_run["elapsed_seconds"],
            },
        },
        "vit_improvement_vs_cnn_percent": 100.0
        * (cnn_metrics["mean_error_m"] - vit_metrics["mean_error_m"])
        / cnn_metrics["mean_error_m"],
        "vit_improvement_vs_naive_percent": 100.0
        * (naive_metrics["mean_error_m"] - vit_metrics["mean_error_m"])
        / naive_metrics["mean_error_m"],
        "hybrid_change_vs_cnn_percent": 100.0
        * (cnn_metrics["mean_error_m"] - hybrid_metrics["mean_error_m"])
        / cnn_metrics["mean_error_m"],
        "hybrid_change_vs_vit_percent": 100.0
        * (vit_metrics["mean_error_m"] - hybrid_metrics["mean_error_m"])
        / vit_metrics["mean_error_m"],
    }
    with (args.output_dir / "comparison_metrics.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
