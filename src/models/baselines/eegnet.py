"""
EEGNet baseline model (Lawhern et al., 2018).

Compact CNN designed specifically for EEG with depthwise and separable convolutions.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """EEGNet: Compact CNN for EEG-based BCIs.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    n_samples : int
        Number of time samples per trial.
    n_classes : int
        Number of output classes.
    F1 : int
        Number of temporal filters.
    F2 : int
        Number of pointwise filters.
    D : int
        Depth multiplier for depthwise convolution.
    kernel_length : int
        Length of temporal convolution kernel.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        n_channels: int = 9,
        n_samples: int = 500,
        n_classes: int = 11,
        F1: int = 8,
        F2: int = 16,
        D: int = 2,
        kernel_length: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()

        # Block 1: Temporal convolution + Depthwise spatial convolution
        self.block1 = nn.Sequential(
            # Temporal convolution
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            # Depthwise spatial convolution
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        # Block 2: Separable convolution
        self.block2 = nn.Sequential(
            # Depthwise temporal convolution
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            # Pointwise convolution
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            dummy = self.block1(dummy)
            dummy = self.block2(dummy)
            flat_size = dummy.numel()

        # Classifier
        self.classifier = nn.Linear(flat_size, n_classes)

    def forward(self, x: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            EEG input, shape (batch, C, T).

        Returns
        -------
        dict with 'cls_logits': (batch, n_classes)
        """
        # Add channel dim: (batch, 1, C, T)
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(1)
        logits = self.classifier(x)
        return {"cls_logits": logits}
