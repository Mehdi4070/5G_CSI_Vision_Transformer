#!/usr/bin/env python3
"""Train and evaluate the compact CSI Vision Transformer."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.train_cnn import (  # noqa: E402
    choose_device,
    evaluate_model,
    load_config,
    make_loader,
    project_path,
    save_cdf_plot,
    save_history_plot,
    save_prediction_plot,
    serializable_evaluation,
    set_reproducibility,
)
from src.metrics import localization_errors, mean_position_baseline  # noqa: E402
from src.models.hybrid import CSILocalizationHybrid  # noqa: E402
from src.models.vit import CSILocalizationViT  # noqa: E402
from src.torch_dataset import CSILocalizationDataset  # noqa: E402
from src.training import train_one_epoch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CSI localization ViT.")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "vit_baseline.yaml"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs.")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-validation", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


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
    loaders = {
        name: make_loader(
            dataset,
            batch_size=int(config["data"]["batch_size"]),
            num_workers=int(config["data"]["num_workers"]),
            shuffle=name == "train",
            seed=seed,
            device=device,
        )
        for name, dataset in datasets.items()
    }

    with (data_directory / "preprocessing_metadata.json").open(
        "r", encoding="utf-8"
    ) as stream:
        metadata = json.load(stream)
    target_mean = np.asarray(
        metadata["target_standardization"]["mean"], dtype=np.float32
    )
    target_std = np.asarray(metadata["target_standardization"]["std"], dtype=np.float32)
    positions = np.load(data_directory / "positions_xy.npy", mmap_mode="r")
    train_positions = np.asarray(positions[datasets["train"].indices])
    naive_predictions: dict[str, np.ndarray] = {}
    naive_metrics: dict[str, dict[str, object]] = {}
    for name in ("validation", "test"):
        split_positions = np.asarray(positions[datasets[name].indices])
        predictions, metrics = mean_position_baseline(train_positions, split_positions)
        naive_predictions[name] = predictions
        naive_metrics[name] = metrics

    model_config = config["model"]
    architecture = str(model_config.get("architecture", "vit"))
    common_model_args = {
        "image_size": tuple(int(value) for value in model_config["image_size"]),
        "embedding_dim": int(model_config["embedding_dim"]),
        "depth": int(model_config["depth"]),
        "num_heads": int(model_config["num_heads"]),
        "mlp_ratio": float(model_config["mlp_ratio"]),
        "dropout": float(model_config["dropout"]),
    }
    if architecture == "vit":
        model = CSILocalizationViT(
            patch_size=tuple(int(value) for value in model_config["patch_size"]),
            **common_model_args,
        )
        model_key = "vit"
        model_label = "ViT"
    elif architecture == "hybrid":
        model = CSILocalizationHybrid(
            stem_channels=tuple(int(value) for value in model_config["stem_channels"]),
            **common_model_args,
        )
        model_key = "hybrid"
        model_label = "CNN-Transformer"
    else:
        raise ValueError(f"Unknown transformer architecture: {architecture}")
    model = model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    sample_features, _, _ = next(iter(loaders["train"]))
    with torch.inference_mode():
        sample_output = model(sample_features[:2].to(device))
    if sample_output.shape != (2, 2) or not torch.isfinite(sample_output).all():
        raise RuntimeError(
            f"{model_label} sanity check failed with output shape {sample_output.shape}"
        )

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
        f"patch_grid={model.patch_grid}; "
        f"train/val/test={len(datasets['train'])}/{len(datasets['validation'])}/"
        f"{len(datasets['test'])}"
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
        history.append(
            {
                "epoch": epoch,
                "train_mse": float(train_mse),
                "validation_mse": validation["normalized_mse"],
                "validation_mean_error_m": validation_mean_error,
                "validation_median_error_m": validation["metrics"][
                    "median_error_m"
                ],
                "learning_rate": current_lr,
            }
        )
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
                    "model_config": model_config,
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
    save_cdf_plot(
        test["errors_m"],
        naive_test_errors,
        output_directory / "test_cdf.png",
        model_label,
    )
    save_prediction_plot(
        test["targets_xy"],
        test["predictions_xy"],
        output_directory / "test_predictions.png",
        model_label,
    )
    np.savez_compressed(
        output_directory / "test_predictions.npz",
        indices=test["indices"],
        targets_xy=test["targets_xy"],
        predictions_xy=test["predictions_xy"],
        errors_m=test["errors_m"],
    )

    results = {
        "model": f"{model_key}_baseline",
        "device": str(device),
        "torch_version": torch.__version__,
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "elapsed_seconds": elapsed,
        "dataset_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "naive_mean_position": naive_metrics,
        model_key: {
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
