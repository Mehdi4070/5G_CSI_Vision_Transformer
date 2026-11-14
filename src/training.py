"""Small, reusable PyTorch regression loops."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    squared_error_sum = 0.0
    num_users = 0
    for features, targets, _ in loader:
        features = features.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(features)
        loss = nn.functional.mse_loss(predictions, targets)
        loss.backward()
        optimizer.step()
        squared_error_sum += float(loss.detach()) * len(features)
        num_users += len(features)
    return squared_error_sum / num_users


@torch.inference_mode()
def predict_normalized(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    global_indices: list[np.ndarray] = []
    squared_error_sum = 0.0
    num_users = 0
    for features, batch_targets, batch_indices in loader:
        features = features.to(device)
        batch_targets_device = batch_targets.to(device)
        batch_predictions = model(features)
        loss = nn.functional.mse_loss(batch_predictions, batch_targets_device)
        squared_error_sum += float(loss) * len(features)
        num_users += len(features)
        predictions.append(batch_predictions.cpu().numpy())
        targets.append(batch_targets.numpy())
        global_indices.append(batch_indices.numpy())
    return (
        np.concatenate(predictions),
        np.concatenate(targets),
        np.concatenate(global_indices),
        squared_error_sum / num_users,
    )

