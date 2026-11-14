#!/usr/bin/env python3
"""Evaluate clean-trained CSI localization models under complex AWGN."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from scripts.train_cnn import evaluate_model  # noqa: E402
from src.csi_preprocessing import csi_to_features, standardize_features  # noqa: E402
from src.models.cnn import CSILocalizationCNN  # noqa: E402
from src.models.hybrid import CSILocalizationHybrid  # noqa: E402
from src.models.vit import CSILocalizationViT  # noqa: E402
from src.noise import add_complex_awgn, measured_snr_db, normalized_complex_noise  # noqa: E402


MODEL_DIRECTORIES = {
    "cnn": PROJECT_ROOT / "outputs" / "cnn_baseline",
    "vit": PROJECT_ROOT / "outputs" / "vit_baseline",
    "hybrid": PROJECT_ROOT / "outputs" / "hybrid_baseline",
}
MODEL_LABELS = {"cnn": "CNN", "vit": "ViT", "hybrid": "CNN-Transformer"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CSI models under AWGN.")
    parser.add_argument("--snr-db", nargs="+", type=float, default=[30, 20, 10, 5, 0])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "noise_robustness",
    )
    return parser.parse_args()


def load_models(device: torch.device) -> dict[str, torch.nn.Module]:
    models: dict[str, torch.nn.Module] = {}
    for name, directory in MODEL_DIRECTORIES.items():
        checkpoint = torch.load(
            directory / "best_model.pt", map_location=device, weights_only=True
        )
        config = checkpoint["model_config"]
        if name == "cnn":
            model = CSILocalizationCNN(
                channels=tuple(int(value) for value in config["channels"]),
                dropout=float(config["dropout"]),
            )
        elif name == "vit":
            model = CSILocalizationViT(
                image_size=tuple(int(value) for value in config["image_size"]),
                patch_size=tuple(int(value) for value in config["patch_size"]),
                embedding_dim=int(config["embedding_dim"]),
                depth=int(config["depth"]),
                num_heads=int(config["num_heads"]),
                mlp_ratio=float(config["mlp_ratio"]),
                dropout=float(config["dropout"]),
            )
        else:
            model = CSILocalizationHybrid(
                image_size=tuple(int(value) for value in config["image_size"]),
                stem_channels=tuple(int(value) for value in config["stem_channels"]),
                embedding_dim=int(config["embedding_dim"]),
                depth=int(config["depth"]),
                num_heads=int(config["num_heads"]),
                mlp_ratio=float(config["mlp_ratio"]),
                dropout=float(config["dropout"]),
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        models[name] = model.to(device).eval()
    return models


def make_loader(
    features: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(features, dtype=np.float32)),
        torch.from_numpy(np.asarray(targets, dtype=np.float32)),
        torch.from_numpy(np.asarray(indices, dtype=np.int64)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def aggregate_metrics(trials: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [key for key in trials[0] if key != "num_users"]
    return {
        "num_users_per_trial": int(trials[0]["num_users"]),
        "num_trials": len(trials),
        "mean": {key: float(np.mean([trial[key] for trial in trials])) for key in keys},
        "std": {key: float(np.std([trial[key] for trial in trials])) for key in keys},
        "trials": trials,
    }


def save_mean_error_plot(
    results: dict[str, dict[str, Any]], snr_values: list[float], path: Path
) -> None:
    labels = ["Clean"] + [f"{value:g}" for value in sorted(snr_values, reverse=True)]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for name, model_results in results.items():
        ordered = ["clean"] + [f"{value:g}" for value in sorted(snr_values, reverse=True)]
        means = [model_results[key]["mean"]["mean_error_m"] for key in ordered]
        stds = [model_results[key]["std"]["mean_error_m"] for key in ordered]
        ax.errorbar(x, means, yerr=stds, marker="o", capsize=3, label=MODEL_LABELS[name])
    ax.set(
        title="Localization robustness to complex AWGN",
        xlabel="CSI condition (SNR [dB])",
        ylabel="Mean localization error [m]",
        xticks=x,
        xticklabels=labels,
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_cdf_plots(
    pooled_errors: dict[str, dict[str, list[float]]],
    snr_values: list[float],
    output_directory: Path,
) -> None:
    ordered = ["clean"] + [f"{value:g}" for value in sorted(snr_values, reverse=True)]
    for name, conditions in pooled_errors.items():
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        for condition in ordered:
            errors = np.sort(np.asarray(conditions[condition]))
            cdf = np.arange(1, len(errors) + 1) / len(errors)
            label = "Clean" if condition == "clean" else f"{condition} dB"
            ax.plot(errors, cdf, label=label)
        ax.set(
            title=f"{MODEL_LABELS[name]} localization-error CDF under AWGN",
            xlabel="Localization error [m]",
            ylabel="CDF",
        )
        ax.grid(alpha=0.3)
        ax.legend()
        fig.savefig(output_directory / f"{name}_noise_cdf.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.repetitions <= 0 or args.batch_size <= 0:
        raise ValueError("repetitions and batch-size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(device)

    data_directory = PROJECT_ROOT / "data" / "processed" / "o1_28_2d"
    with (data_directory / "preprocessing_metadata.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    feature_mean = np.asarray(metadata["feature_standardization"]["mean"], dtype=np.float32)
    feature_std = np.asarray(metadata["feature_standardization"]["std"], dtype=np.float32)
    target_mean = np.asarray(metadata["target_standardization"]["mean"], dtype=np.float32)
    target_std = np.asarray(metadata["target_standardization"]["std"], dtype=np.float32)
    amplitude_floor = float(metadata["configuration"]["features"]["amplitude_floor"])
    with np.load(data_directory / "split_indices.npz", allow_pickle=False) as splits:
        test_indices = np.asarray(splits["test"], dtype=np.int64)
    targets = np.asarray(np.load(data_directory / "targets_normalized.npy", mmap_mode="r")[test_indices])

    raw_path = PROJECT_ROOT / "data" / "raw" / "o1_28_2d" / "csi_raw.npz"
    print(f"Loading clean raw CSI from {raw_path}...")
    with np.load(raw_path, allow_pickle=False) as raw:
        clean_csi = np.asarray(raw["csi"][test_indices], dtype=np.complex64)
    clean_features = standardize_features(
        csi_to_features(clean_csi, amplitude_floor=amplitude_floor),
        mean=feature_mean,
        std=feature_std,
    )
    stored_features = np.load(data_directory / "features.npy", mmap_mode="r")
    max_feature_difference = float(
        np.max(np.abs(clean_features - np.asarray(stored_features[test_indices])))
    )
    if max_feature_difference > 1e-5:
        raise RuntimeError(
            f"Reconstructed clean features differ from stored features by {max_feature_difference}"
        )

    results: dict[str, dict[str, Any]] = {name: {} for name in models}
    pooled_errors: dict[str, dict[str, list[float]]] = {name: {} for name in models}
    clean_loader = make_loader(clean_features, targets, test_indices, args.batch_size)
    for name, model in models.items():
        evaluation = evaluate_model(model, clean_loader, device, target_mean, target_std)
        metrics = evaluation["metrics"]
        results[name]["clean"] = aggregate_metrics([metrics])
        pooled_errors[name]["clean"] = evaluation["errors_m"].tolist()
        print(f"{MODEL_LABELS[name]} clean mean error: {metrics['mean_error_m']:.3f} m")

    rng = np.random.default_rng(args.seed)
    trial_results: dict[str, dict[str, list[dict[str, Any]]]] = {
        name: {f"{snr:g}": [] for snr in args.snr_db} for name in models
    }
    trial_errors: dict[str, dict[str, list[float]]] = {
        name: {f"{snr:g}": [] for snr in args.snr_db} for name in models
    }
    measured_snr: dict[str, list[float]] = {f"{snr:g}": [] for snr in args.snr_db}
    for repetition in range(args.repetitions):
        unit_noise = normalized_complex_noise(clean_csi.shape, rng)
        for snr in args.snr_db:
            key = f"{snr:g}"
            noisy_csi = add_complex_awgn(clean_csi, snr_db=snr, unit_noise=unit_noise)
            measured_snr[key].append(float(measured_snr_db(clean_csi, noisy_csi).mean()))
            noisy_features = standardize_features(
                csi_to_features(noisy_csi, amplitude_floor=amplitude_floor),
                mean=feature_mean,
                std=feature_std,
            )
            loader = make_loader(noisy_features, targets, test_indices, args.batch_size)
            for name, model in models.items():
                evaluation = evaluate_model(model, loader, device, target_mean, target_std)
                trial_results[name][key].append(evaluation["metrics"])
                trial_errors[name][key].extend(evaluation["errors_m"].tolist())
            print(
                f"Repetition {repetition + 1}/{args.repetitions}, SNR={snr:g} dB complete"
            )
            del noisy_csi, noisy_features, loader
            gc.collect()

    for name in models:
        for snr in args.snr_db:
            key = f"{snr:g}"
            results[name][key] = aggregate_metrics(trial_results[name][key])
            pooled_errors[name][key] = trial_errors[name][key]

    clean_reference: dict[str, float] = {}
    for name, directory in MODEL_DIRECTORIES.items():
        with (directory / "metrics.json").open(encoding="utf-8") as stream:
            existing = json.load(stream)
        clean_reference[name] = float(existing[name]["test"]["metrics"]["mean_error_m"])
        observed = results[name]["clean"]["mean"]["mean_error_m"]
        if abs(observed - clean_reference[name]) > 1e-5:
            raise RuntimeError(f"Clean {name} metric does not reproduce the stored result")

    save_mean_error_plot(results, args.snr_db, args.output_dir / "mean_error_vs_snr.png")
    save_cdf_plots(pooled_errors, args.snr_db, args.output_dir)
    summary = {
        "protocol": {
            "noise": "per-user complex AWGN added before feature extraction",
            "snr_db": args.snr_db,
            "repetitions": args.repetitions,
            "seed": args.seed,
            "models_trained_on": "clean CSI only",
            "normalization": "clean training statistics only",
            "num_test_users": int(len(test_indices)),
            "device": str(device),
            "max_clean_feature_reconstruction_difference": max_feature_difference,
            "measured_snr_db_mean": measured_snr,
        },
        "clean_reference_mean_error_m": clean_reference,
        "results": results,
    }
    with (args.output_dir / "noise_robustness.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from None
