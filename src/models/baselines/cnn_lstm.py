"""
CNN-LSTM baseline — CNN feature extractor + LSTM temporal model.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """CNN feature extractor followed by LSTM temporal model.

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
    lstm_hidden : int
        LSTM hidden dimension.
    lstm_layers : int
        Number of LSTM layers.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        n_channels: int = 9,
        n_samples: int = 500,
        n_classes: int = 11,
        cnn_filters: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        # CNN feature extractor (temporal convolutions)
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

        # LSTM temporal model
        self.lstm = nn.LSTM(
            input_size=cnn_filters * 2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            EEG input, shape (batch, C, T).
        """
        # CNN: (batch, C, T) → (batch, cnn_out, T')
        features = self.cnn(x)

        # Transpose for LSTM: (batch, T', cnn_out)
        features = features.transpose(1, 2)

        # LSTM: take last hidden state
        lstm_out, (h_n, _) = self.lstm(features)
        last_hidden = h_n[-1]  # (batch, lstm_hidden)

        return {"cls_logits": self.classifier(last_hidden)}
