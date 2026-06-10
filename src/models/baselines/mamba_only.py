"""
Mamba-only baseline — Mamba temporal encoder without attention or convolution.

Used to isolate the contribution of the Mamba SSM component.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.mamba_module import MambaModule


class MambaOnlyModel(nn.Module):
    """Mamba-only EEG model: SSM temporal encoder without attention.

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
        Number of Mamba layers.
    d_state : int
        SSM state dimension.
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
        d_state: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(n_samples, d_model)

        # Mamba encoder
        self.mamba = MambaModule(
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=4,
            expand=2,
            dropout=dropout,
        )

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
        # Project: (batch, C, T) → (batch, C, d_model)
        x = self.input_proj(x)

        # Mamba: (batch, C, d_model) → (batch, C, d_model)
        x = self.mamba(x)

        # Mean pooling
        x = x.mean(dim=1)  # (batch, d_model)

        return {"cls_logits": self.classifier(x)}
