"""
Vanilla Transformer baseline — standard Transformer encoder (no convolution module).

Used to demonstrate that the Conformer's convolution module adds value.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaTransformer(nn.Module):
    """Standard Transformer encoder for EEG classification.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    n_samples : int
        Number of time samples.
    n_classes : int
        Number of output classes.
    d_model : int
        Model dimension.
    n_layers : int
        Number of Transformer layers.
    n_heads : int
        Number of attention heads.
    d_ff : int
        Feed-forward dimension.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        n_channels: int = 9,
        n_samples: int = 500,
        n_classes: int = 11,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Input projection: (C, T) → (C, d_model) per channel as token
        self.input_proj = nn.Linear(n_samples, d_model)

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, n_channels, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            EEG input, shape (batch, C, T).
        """
        # Project each channel's time series to d_model
        x = self.input_proj(x)  # (batch, C, d_model)
        x = x + self.pos_embed

        # Transformer encoder
        x = self.encoder(x)  # (batch, C, d_model)
        x = self.norm(x)

        # Mean pooling over channels
        x = x.mean(dim=1)  # (batch, d_model)

        return {"cls_logits": self.classifier(x)}
