"""Lightweight convolutional baseline for CSI position regression."""

from __future__ import annotations

import torch
from torch import nn


class CSILocalizationCNN(nn.Module):
    """Regress normalized XY coordinates from a three-channel CSI image."""

    def __init__(
        self,
        *,
        input_channels: int = 3,
        channels: tuple[int, ...] = (16, 32, 64, 96),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("At least one convolutional channel is required")

        blocks: list[nn.Module] = []
        in_channels = input_channels
        for index, out_channels in enumerate(channels):
            kernel_size = 5 if index == 0 else 3
            blocks.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.pool(self.features(inputs)))

