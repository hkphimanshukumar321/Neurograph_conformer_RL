"""
CNN-BiGRU baseline — CNN feature extractor + Bidirectional GRU.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CNNBiGRU(nn.Module):
    """CNN + Bidirectional GRU for EEG decoding.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    n_samples : int
        Number of time samples.
    n_classes : int
        Number of output classes.
    cnn_filters : int
        Number of CNN filters.
    gru_hidden : int
        GRU hidden dimension.
    gru_layers : int
        Number of GRU layers.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        n_channels: int = 9,
        n_samples: int = 500,
        n_classes: int = 11,
        cnn_filters: int = 64,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, cnn_filters, kernel_size=25, padding=12),
            nn.BatchNorm1d(cnn_filters),
            nn.ELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_filters, cnn_filters * 2, kernel_size=10, padding=5),
            nn.BatchNorm1d(cnn_filters * 2),
            nn.ELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            input_size=cnn_filters * 2,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hidden * 2, n_classes),  # *2 for bidirectional
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.cnn(x)
        features = features.transpose(1, 2)
        gru_out, h_n = self.gru(features)

        # Concatenate forward and backward final hidden states
        last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)

        return {"cls_logits": self.classifier(last_hidden)}
