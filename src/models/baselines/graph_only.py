"""
Graph-only baseline — GATv2 spatial encoder + MLP head (no temporal modeling).

Used to isolate the contribution of spatial graph modeling.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.graph_encoder import GraphSpatialEncoder


class GraphOnlyModel(nn.Module):
    """Graph-only EEG model: spatial encoder without temporal modeling.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    n_samples : int
        Number of time samples (flattened as input features).
    n_classes : int
        Number of output classes.
    d_model : int
        Graph encoder output dimension.
    n_graph_layers : int
        Number of GATv2 layers.
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        n_channels: int = 9,
        n_samples: int = 500,
        n_classes: int = 11,
        d_model: int = 128,
        n_graph_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.graph_encoder = GraphSpatialEncoder(
            d_in=n_samples,
            d_model=d_model,
            n_layers=n_graph_layers,
            n_heads=n_heads,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            EEG input, shape (batch, C, T). T is used as node feature dim.
        adj : torch.Tensor
            Adjacency matrix, shape (C, C).
        """
        # x: (batch, C, T) — each channel's time series is the node feature
        if adj is None:
            # Default: fully connected graph
            n_ch = x.size(1)
            adj = torch.ones(n_ch, n_ch, device=x.device)

        h = self.graph_encoder(x, adj)  # (batch, C, d_model)

        # Mean pooling over nodes
        pooled = h.mean(dim=1)  # (batch, d_model)

        return {"cls_logits": self.classifier(pooled)}
