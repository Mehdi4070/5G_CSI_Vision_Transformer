#!/usr/bin/env python3
"""Train and evaluate the CNN baseline for CSI localization."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))
from src.metrics import (  # noqa: E402
    localization_errors,
    localization_metrics,
    mean_position_baseline,
)
from src.models.cnn import CSILocalizationCNN  # noqa: E402
from src.torch_dataset import CSILocalizationDataset  # noqa: E402
from src.training import predict_normalized, train_one_epoch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CSI localization CNN baseline.")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "cnn_baseline.yaml"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs.")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-validation", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required = {"data", "model", "training", "output"}
    missing = required.difference(config or {})
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    return config


def set_reproducibility(seed: int, cpu_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def make_loader(
    dataset: CSILocalizationDataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def to_physical_coordinates(
    normalized: np.ndarray, target_mean: np.ndarray, target_std: np.ndarray
) -> np.ndarray:
    return normalized * target_std + target_mean


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> dict[str, Any]:
    predictions_norm, targets_norm, indices, mse = predict_normalized(model, loader, device)
    predictions_xy = to_physical_coordinates(predictions_norm, target_mean, target_std)
    targets_xy = to_physical_coordinates(targets_norm, target_mean, target_std)
    return {
        "normalized_mse": float(mse),
        "metrics": localization_metrics(predictions_xy, targets_xy),
        "predictions_xy": predictions_xy,
        "targets_xy": targets_xy,
        "indices": indices,
        "errors_m": localization_errors(predictions_xy, targets_xy),
    }


def save_history_plot(history: list[dict[str, float]], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].plot(epochs, [row["train_mse"] for row in history], label="train")
    axes[0].plot(epochs, [row["validation_mse"] for row in history], label="validation")
    axes[0].set(title="Normalized coordinate loss", xlabel="Epoch", ylabel="MSE")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].plot(epochs, [row["validation_mean_error_m"] for row in history])
    axes[1].set(title="Validation localization error", xlabel="Epoch", ylabel="Mean error [m]")
    axes[1].grid(alpha=0.3)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_cdf_plot(
    model_errors: np.ndarray,
    naive_errors: np.ndarray,
    path: Path,
    model_label: str = "CNN",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for errors, label in (
        (naive_errors, "Mean-position baseline"),
        (model_errors, model_label),
    ):
        sorted_errors = np.sort(errors)
        cdf = np.arange(1, len(errors) + 1) / len(errors)
        ax.plot(sorted_errors, cdf, label=label)
    ax.set(title="Test localization-error CDF", xlabel="Error [m]", ylabel="CDF")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_prediction_plot(
    targets: np.ndarray,
    predictions: np.ndarray,
    path: Path,
    model_label: str = "CNN",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax.scatter(targets[:, 0], targets[:, 1], s=9, alpha=0.35, label="True")
    ax.scatter(predictions[:, 0], predictions[:, 1], s=9, alpha=0.35, label=model_label)
    ax.set(title=f"{model_label} test predictions", xlabel="x [m]", ylabel="y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def serializable_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        "normalized_mse": evaluation["normalized_mse"],
        "metrics": evaluation["metrics"],
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    training_config = config["training"]
    seed = int(training_config["seed"])
    set_reproducibility(seed, int(training_config["cpu_threads"]))
    device = choose_device(str(training_config["device"]))
    data_directory = project_path(config["data"]["directory"])
    output_directory = project_path(args.output_dir or config["output"]["directory"])
    output_directory.mkdir(parents=True, exist_ok=True)

    datasets = {
        "train": CSILocalizationDataset(data_directory, "train", limit=args.limit_train),
        "validation": CSILocalizationDataset(
            data_directory, "validation", limit=args.limit_validation
        ),
        "test": CSILocalizationDataset(data_directory, "test", limit=args.limit_test),
    }
    batch_size = int(config["data"]["batch_size"])
    num_workers = int(config["data"]["num_workers"])
    loaders = {
        name: make_loader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=name == "train",
            seed=seed,
            device=device,
        )
        for name, dataset in datasets.items()
    }

    with (data_directory / "preprocessing_metadata.json").open("r", encoding="utf-8") as stream:
        preprocessing_metadata = json.load(stream)
    target_mean = np.asarray(
        preprocessing_metadata["target_standardization"]["mean"], dtype=np.float32
    )
    target_std = np.asarray(
        preprocessing_metadata["target_standardization"]["std"], dtype=np.float32
    )
    positions = np.load(data_directory / "positions_xy.npy", mmap_mode="r")

    train_positions = np.asarray(positions[datasets["train"].indices])
    naive_predictions: dict[str, np.ndarray] = {}
    naive_metrics: dict[str, dict[str, Any]] = {}
    for name in ("validation", "test"):
        split_positions = np.asarray(positions[datasets[name].indices])
        predictions, metrics = mean_position_baseline(train_positions, split_positions)
        naive_predictions[name] = predictions
        naive_metrics[name] = metrics

    model = CSILocalizationCNN(
        channels=tuple(int(value) for value in config["model"]["channels"]),
        dropout=float(config["model"]["dropout"]),
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    sample_features, _, _ = next(iter(loaders["train"]))
    with torch.inference_mode():
        sample_output = model(sample_features[:2].to(device))
    if sample_output.shape != (2, 2) or not torch.isfinite(sample_output).all():
        raise RuntimeError(f"CNN sanity check failed with output shape {sample_output.shape}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training_config["scheduler_factor"]),
        patience=int(training_config["scheduler_patience"]),
        min_lr=float(training_config["minimum_learning_rate"]),
    )

    max_epochs = int(args.epochs or training_config["max_epochs"])
    early_patience = int(training_config["early_stopping_patience"])
    min_delta = float(training_config["early_stopping_min_delta_m"])
    history: list[dict[str, float]] = []
    best_mean_error = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_path = output_directory / "best_model.pt"
    started = time.perf_counter()

    print(
        f"Device={device}; parameters={parameter_count:,}; "
        f"train/val/test={len(datasets['train'])}/{len(datasets['validation'])}/{len(datasets['test'])}"
    )
    print(f"Naive test mean error: {naive_metrics['test']['mean_error_m']:.3f} m")
    for epoch in range(1, max_epochs + 1):
        train_mse = train_one_epoch(model, loaders["train"], optimizer, device)
        validation = evaluate_model(
            model, loaders["validation"], device, target_mean, target_std
        )
        validation_mean_error = validation["metrics"]["mean_error_m"]
        scheduler.step(validation_mean_error)
        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_mse": float(train_mse),
            "validation_mse": validation["normalized_mse"],
            "validation_mean_error_m": validation_mean_error,
            "validation_median_error_m": validation["metrics"]["median_error_m"],
            "learning_rate": current_lr,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}: train_mse={train_mse:.5f}, "
            f"val_mean={validation_mean_error:.3f} m, lr={current_lr:.2e}"
        )

        if validation_mean_error < best_mean_error - min_delta:
            best_mean_error = validation_mean_error
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": validation["metrics"],
                    "model_config": config["model"],
                    "target_mean": target_mean.tolist(),
                    "target_std": target_std.tolist(),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation = evaluate_model(model, loaders["validation"], device, target_mean, target_std)
    test = evaluate_model(model, loaders["test"], device, target_mean, target_std)
    elapsed = time.perf_counter() - started

    naive_test_errors = localization_errors(
        naive_predictions["test"], np.asarray(positions[datasets["test"].indices])
    )
    save_history_plot(history, output_directory / "training_curves.png")
    save_cdf_plot(test["errors_m"], naive_test_errors, output_directory / "test_cdf.png")
    save_prediction_plot(
        test["targets_xy"], test["predictions_xy"], output_directory / "test_predictions.png"
    )
    np.savez_compressed(
        output_directory / "test_predictions.npz",
        indices=test["indices"],
        targets_xy=test["targets_xy"],
        predictions_xy=test["predictions_xy"],
        errors_m=test["errors_m"],
    )

    results = {
        "model": "cnn_baseline",
        "device": str(device),
        "torch_version": torch.__version__,
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "elapsed_seconds": elapsed,
        "dataset_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "naive_mean_position": naive_metrics,
        "cnn": {
            "validation": serializable_evaluation(validation),
            "test": serializable_evaluation(test),
        },
        "configuration": config,
        "cli_limits": {
            "train": args.limit_train,
            "validation": args.limit_validation,
            "test": args.limit_test,
            "epochs_override": args.epochs,
        },
    }
    with (output_directory / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
    with (output_directory / "history.json").open("w", encoding="utf-8") as stream:
        json.dump(history, stream, indent=2)

    print("\nFinal test metrics:")
    print(json.dumps(test["metrics"], indent=2))
    print(f"Best epoch: {best_epoch}; elapsed: {elapsed:.1f} s")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from None
