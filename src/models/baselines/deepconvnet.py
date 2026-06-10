"""
DeepConvNet baseline (Schirrmeister et al., 2017).

Deeper CNN architecture for EEG classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DeepConvNet(nn.Module):
    """DeepConvNet for EEG decoding.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    n_samples : int
        Number of time samples per trial.
    n_classes : int
        Number of output classes.
    n_filters_1 : int
        Number of filters in first convolution layer.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        n_channels: int = 9,
        n_samples: int = 500,
        n_classes: int = 11,
        n_filters_1: int = 25,
        dropout: float = 0.5,
    ):
        super().__init__()

        # Block 1: Temporal + Spatial convolution
        self.block1 = nn.Sequential(
            nn.Conv2d(1, n_filters_1, (1, 10), bias=False),
            nn.Conv2d(n_filters_1, n_filters_1, (n_channels, 1), bias=False),
            nn.BatchNorm2d(n_filters_1),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout(dropout),
        )

        # Blocks 2–4: Progressive filtering
        self.block2 = self._make_block(n_filters_1, 50, dropout)
        self.block3 = self._make_block(50, 100, dropout)
        self.block4 = self._make_block(100, 200, dropout)

        # Calculate output size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            dummy = self.block1(dummy)
            dummy = self.block2(dummy)
            dummy = self.block3(dummy)
            dummy = self.block4(dummy)
            flat_size = dummy.numel()

        self.classifier = nn.Linear(flat_size, n_classes)

    @staticmethod
    def _make_block(in_ch: int, out_ch: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (1, 10), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.unsqueeze(1)  # (batch, 1, C, T)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.flatten(1)
        return {"cls_logits": self.classifier(x)}
