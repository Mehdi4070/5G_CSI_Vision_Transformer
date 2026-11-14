"""Compact Vision Transformer for CSI-based position regression."""

from __future__ import annotations

import torch
from torch import nn


class CSILocalizationViT(nn.Module):
    """Regress normalized XY coordinates from antenna x subcarrier CSI maps."""

    def __init__(
        self,
        *,
        image_size: tuple[int, int] = (64, 64),
        patch_size: tuple[int, int] = (8, 8),
        input_channels: int = 3,
        embedding_dim: int = 96,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if any(size <= 0 for size in (*image_size, *patch_size)):
            raise ValueError("Image and patch dimensions must be positive")
        if image_size[0] % patch_size[0] or image_size[1] % patch_size[1]:
            raise ValueError("Image dimensions must be divisible by patch dimensions")
        if embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_grid = (
            image_size[0] // patch_size[0],
            image_size[1] // patch_size[1],
        )
        self.patch_embedding = nn.Conv2d(
            input_channels,
            embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Antenna and subcarrier positions are learned independently, then added.
        self.antenna_position = nn.Parameter(
            torch.zeros(1, self.patch_grid[0], 1, embedding_dim)
        )
        self.subcarrier_position = nn.Parameter(
            torch.zeros(1, 1, self.patch_grid[1], embedding_dim)
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.class_position = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=int(embedding_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=depth, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.regressor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 2),
        )
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        nn.init.trunc_normal_(self.antenna_position, std=0.02)
        nn.init.trunc_normal_(self.subcarrier_position, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.class_position, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if tuple(inputs.shape[-2:]) != self.image_size:
            raise ValueError(
                f"Expected CSI maps with size {self.image_size}, got {tuple(inputs.shape[-2:])}"
            )
        patches = self.patch_embedding(inputs).permute(0, 2, 3, 1)
        positions = self.antenna_position + self.subcarrier_position
        tokens = (patches + positions).flatten(1, 2)
        class_token = self.class_token.expand(len(inputs), -1, -1) + self.class_position
        tokens = torch.cat((class_token, tokens), dim=1)
        encoded = self.encoder(self.input_dropout(tokens))
        return self.regressor(self.norm(encoded[:, 0]))
