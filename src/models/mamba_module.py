"""
Mamba Module — Selective State Space Model for long-range sequence modeling.

Placed AFTER the Conformer encoder to capture long-range sequential dependencies
with O(L) complexity (vs O(L²) for attention). Most useful for sentence-level
tasks where EEG is segmented into multiple windows.

Uses the mamba-ssm package on Linux; provides a fallback S4-like implementation
for non-Linux platforms.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaModule(nn.Module):
    """Mamba sequential refinement module.

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_layers : int
        Number of Mamba blocks.
    d_state : int
        SSM state dimension.
    d_conv : int
        Local convolution width.
    expand : int
        Expansion factor for inner dimension.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        for _ in range(n_layers):
            self.layers.append(
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
            )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence, shape (batch, L, d_model).

        Returns
        -------
        torch.Tensor
            Refined sequence, shape (batch, L, d_model).
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class MambaBlock(nn.Module):
    """Single Mamba block with selective scan.

    Implements a simplified version of the Mamba architecture:
    - Input projection (expand)
    - 1D convolution (local context)
    - Selective SSM
    - Output projection

    Falls back to a gated S4-like block if mamba-ssm is not available.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state

        self.norm = nn.LayerNorm(d_model)

        # Try to use the official mamba-ssm package
        self._use_mamba_ssm = False
        try:
            from mamba_ssm import Mamba

            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self._use_mamba_ssm = True
        except ImportError:
            # Fallback: simplified SSM implementation
            self._build_fallback(d_model, d_state, d_conv, expand, dropout)

        self.dropout = nn.Dropout(dropout)

    def _build_fallback(
        self, d_model: int, d_state: int, d_conv: int, expand: int, dropout: float
    ):
        """Build a simplified gated SSM block as fallback."""
        d_inner = d_model * expand

        # Input projection (split into two paths for gating)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)

        # Local convolution
        self.conv1d = nn.Conv1d(
            d_inner, d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner,
        )

        # SSM parameters (simplified: learned per-dimension)
        self.A_log = nn.Parameter(torch.randn(d_inner, d_state))
        self.D = nn.Parameter(torch.ones(d_inner))

        # Selective projections (input-dependent)
        self.x_proj = nn.Linear(d_inner, d_state * 2, bias=False)  # B, C
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x
        x = self.norm(x)

        if self._use_mamba_ssm:
            x = self.mamba(x)
        else:
            x = self._forward_fallback(x)

        return residual + self.dropout(x)

    def _forward_fallback(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback SSM forward pass with proper selective scan recurrence.

        Implements a simplified but functional selective state-space model:
        1. Input projection → split into two paths (for gating)
        2. 1D causal convolution (local context)
        3. Input-dependent discretization: compute B, C, dt from input
        4. Sequential scan: h[t] = A_bar * h[t-1] + B_bar * x[t]; y[t] = C * h[t]
        5. Skip connection via D parameter
        6. Output gating and projection
        """
        batch, seq_len, d = x.shape

        # Input projection
        xz = self.in_proj(x)  # (batch, L, 2*d_inner)
        x_inner, z = xz.chunk(2, dim=-1)  # each (batch, L, d_inner)

        # 1D convolution (local context)
        x_conv = x_inner.transpose(1, 2)  # (batch, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # causal: truncate
        x_conv = x_conv.transpose(1, 2)  # (batch, L, d_inner)
        x_conv = F.silu(x_conv)

        # Input-dependent SSM parameters
        # x_proj produces B and C (each d_state-dimensional)
        bc = self.x_proj(x_conv)  # (batch, L, 2*d_state)
        B, C = bc.chunk(2, dim=-1)  # each (batch, L, d_state)

        # Input-dependent step size (dt)
        dt = F.softplus(self.dt_proj(x_conv))  # (batch, L, d_inner), positive

        # Discretize continuous SSM: A_bar = exp(A * dt), B_bar = dt * B
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state), negative for stability

        # Sequential scan
        y = torch.zeros_like(x_conv)  # (batch, L, d_inner)
        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)

        for t in range(seq_len):
            dt_t = dt[:, t, :]  # (batch, d_inner)
            B_t = B[:, t, :]    # (batch, d_state)
            C_t = C[:, t, :]    # (batch, d_state)
            x_t = x_conv[:, t, :]  # (batch, d_inner)

            # Discretize: A_bar = exp(A * dt), where A is (d_inner, d_state)
            # dt_t is (batch, d_inner) → expand for broadcasting
            A_bar = torch.exp(A.unsqueeze(0) * dt_t.unsqueeze(-1))  # (batch, d_inner, d_state)

            # B_bar = dt * B (outer product-like)
            B_bar = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)  # (batch, d_inner, d_state)

            # State update: h = A_bar * h + B_bar * x
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)  # (batch, d_inner, d_state)

            # Output: y = C * h (summed over state dimension)
            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1)  # (batch, d_inner)
            y[:, t, :] = y_t

        # Skip connection
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)

        # Gate with second path
        y = y * F.silu(z)

        # Output projection
        return self.out_proj(y)
