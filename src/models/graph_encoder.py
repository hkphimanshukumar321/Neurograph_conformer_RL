"""
Graph Spatial Encoder — GATv2 for inter-electrode spatial dependency modeling.

Models the spatial relationships between EEG channels using graph attention.
Supports variable electrode montages via position-based adjacency.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSpatialEncoder(nn.Module):
    """GATv2-based spatial encoder for EEG electrode graphs.

    Parameters
    ----------
    d_in : int
        Input feature dimension per node (channel).
    d_model : int
        Output feature dimension per node.
    n_layers : int
        Number of GATv2 layers.
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    adjacency_type : str
        How to compute adjacency: 'distance', 'correlation', 'hybrid'.
    sigma : float or None
        Gaussian kernel sigma for distance-based adjacency. None = auto (median distance).
    """

    def __init__(
        self,
        d_in: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        adjacency_type: str = "distance",
        sigma: float | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.adjacency_type = adjacency_type
        self.sigma = sigma

        # Input projection
        self.input_proj = nn.Linear(d_in, d_model)

        # GATv2 layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                GATv2Layer(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                )
            )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def compute_adjacency(
        self,
        positions: torch.Tensor,
        data: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute adjacency matrix from electrode positions.

        Parameters
        ----------
        positions : torch.Tensor
            3D electrode positions (MNI coords), shape (C, 3).
        data : torch.Tensor, optional
            EEG data for correlation-based adjacency, shape (C, T).

        Returns
        -------
        torch.Tensor
            Adjacency matrix, shape (C, C).
        """
        if self.adjacency_type == "distance":
            return self._distance_adjacency(positions)
        elif self.adjacency_type == "correlation" and data is not None:
            return self._correlation_adjacency(data)
        elif self.adjacency_type == "hybrid" and data is not None:
            a_dist = self._distance_adjacency(positions)
            a_corr = self._correlation_adjacency(data)
            return 0.5 * a_dist + 0.5 * a_corr
        else:
            return self._distance_adjacency(positions)

    def _distance_adjacency(self, positions: torch.Tensor) -> torch.Tensor:
        """Gaussian kernel adjacency from 3D positions."""
        # Pairwise Euclidean distance
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # (C, C, 3)
        dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)  # (C, C)

        # Auto sigma: median inter-electrode distance
        sigma = self.sigma
        if sigma is None or sigma <= 0:
            sigma = float(torch.median(dist[dist > 0]).item())

        # Gaussian kernel
        adj = torch.exp(-dist ** 2 / (2 * sigma ** 2))
        # Remove self-loops (optional, can keep for residual)
        # adj.fill_diagonal_(0)
        return adj

    def _correlation_adjacency(self, data: torch.Tensor) -> torch.Tensor:
        """Correlation-based adjacency from EEG data."""
        # data: (C, T)
        data_centered = data - data.mean(dim=-1, keepdim=True)
        norms = data_centered.norm(dim=-1, keepdim=True) + 1e-8
        data_normed = data_centered / norms
        corr = torch.mm(data_normed, data_normed.t())  # (C, C)
        return torch.abs(corr)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through graph spatial encoder.

        Parameters
        ----------
        x : torch.Tensor
            Node features, shape (batch, C, d_in).
        adj : torch.Tensor
            Adjacency matrix, shape (C, C) or (batch, C, C).

        Returns
        -------
        torch.Tensor
            Encoded node features, shape (batch, C, d_model).
        """
        x = self.input_proj(x)  # (batch, C, d_model)

        for layer in self.layers:
            x = layer(x, adj)

        x = self.norm(x)
        return x


class GATv2Layer(nn.Module):
    """Single GATv2 (Graph Attention Network v2) layer.

    GATv2 fixes the expressiveness limitation of original GAT by applying
    the learned attention function to the concatenation of transformed
    source and target features (dynamic attention).

    Parameters
    ----------
    d_model : int
        Feature dimension.
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Linear transformations for source and target
        self.W_src = nn.Linear(d_model, d_model, bias=False)
        self.W_tgt = nn.Linear(d_model, d_model, bias=False)

        # Attention vector (per head)
        self.attn = nn.Parameter(torch.randn(n_heads, self.d_head))
        nn.init.xavier_uniform_(self.attn.unsqueeze(0))

        # Output
        self.W_out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node features, shape (batch, N, d_model).
        adj : torch.Tensor
            Adjacency matrix, shape (N, N) or (batch, N, N).

        Returns
        -------
        torch.Tensor
            Updated features, shape (batch, N, d_model).
        """
        residual = x
        batch, n_nodes, d = x.shape

        # Transform
        src = self.W_src(x)  # (batch, N, d_model)
        tgt = self.W_tgt(x)  # (batch, N, d_model)

        # Reshape for multi-head: (batch, N, n_heads, d_head)
        src = src.view(batch, n_nodes, self.n_heads, self.d_head)
        tgt = tgt.view(batch, n_nodes, self.n_heads, self.d_head)

        # GATv2 attention: a^T · LeakyReLU(W_src·h_i + W_tgt·h_j)
        # For all pairs (i, j): (batch, N, N, n_heads, d_head)
        combined = src.unsqueeze(2) + tgt.unsqueeze(1)  # (batch, N, N, n_heads, d_head)
        combined = F.leaky_relu(combined, negative_slope=0.2)

        # Attention scores: (batch, N, N, n_heads)
        attn_scores = (combined * self.attn.unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(dim=-1)

        # Mask using adjacency (prevent attention to non-connected nodes)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).unsqueeze(-1)  # (1, N, N, 1)
        elif adj.dim() == 3:
            adj = adj.unsqueeze(-1)  # (batch, N, N, 1)

        attn_scores = attn_scores.masked_fill(adj == 0, float("-inf"))

        # Softmax over neighbors
        attn_weights = F.softmax(attn_scores, dim=2)  # (batch, N, N, n_heads)
        attn_weights = self.dropout(attn_weights)

        # Aggregate: weighted sum of target features
        # (batch, N, N, n_heads, 1) * (batch, 1, N, n_heads, d_head)
        tgt_expanded = tgt.unsqueeze(1).expand(-1, n_nodes, -1, -1, -1)
        out = (attn_weights.unsqueeze(-1) * tgt_expanded).sum(dim=2)
        # (batch, N, n_heads, d_head) → (batch, N, d_model)
        out = out.reshape(batch, n_nodes, -1)

        out = self.W_out(out)
        out = self.dropout(out)

        # Residual + LayerNorm
        x = self.norm(residual + out)

        # FFN with residual
        x = self.norm2(x + self.ffn(x))

        return x
