"""
Lightweight EEG front-end — Ghost convolutions, Depthwise ASPP, Coordinate Attention.

Purpose:
    Extracts local time-frequency features from wavelet/filter-bank EEG tensors
    *before* the graph spatial encoder. All 2D convolutions are applied
    electrode-wise (reshape [B, N, F, T] → [B*N, 1, F, T]) so that
    inter-electrode mixing is deferred entirely to the graph encoder.

Input:  x_tf   [B, N, F, T]  — wavelet/filter-bank time-frequency features
Output: x_out  [B, N, D, T]  — local features, electrode identity + time preserved

Modules:
    GhostConv2D           — lightweight projection via cheap depthwise ghosts
    DepthwiseASPP2D       — multi-scale temporal receptive field (time-axis dilations)
    CoordinateAttention2D — recalibrates frequency-time features with positional info
    EEGGhostDWASPPCASFrontEnd — composite block with ablation-friendly switches
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# GhostConv2D: lightweight feature projection for low-parameter EEG extraction
# ---------------------------------------------------------------------------

class GhostConv2D(nn.Module):
    """Ghost convolution for EEG time-frequency tensors.

    Generates ``out_channels`` feature maps using only ``out_channels // ratio``
    intrinsic convolutions; the remaining maps are produced by cheap depthwise
    linear operations on the intrinsic features.

    Parameters
    ----------
    in_channels : int
        Input channels (typically 1 for electrode-wise processing).
    out_channels : int
        Total output channels.
    kernel_size : tuple[int, int]
        Kernel size for the primary convolution.
    ratio : int
        Ghost ratio — controls intrinsic vs. ghost split. Default 2.
    stride : tuple[int, int]
        Stride of the primary convolution.
    norm : str
        Normalization type: ``'batchnorm'`` or ``'layernorm'``.
    activation : str
        Activation type: ``'gelu'``, ``'relu'``, ``'silu'``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 64,
        kernel_size: tuple[int, int] = (3, 3),
        ratio: int = 2,
        stride: tuple[int, int] = (1, 1),
        norm: str = "batchnorm",
        activation: str = "gelu",
    ):
        super().__init__()
        assert out_channels % ratio == 0, "out_channels must be divisible by ratio"
        intrinsic_ch = out_channels // ratio
        ghost_ch = out_channels - intrinsic_ch  # remaining maps

        pad = (kernel_size[0] // 2, kernel_size[1] // 2)

        # Primary (intrinsic) convolution — standard conv
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, intrinsic_ch, kernel_size, stride, pad, bias=False),
            _make_norm(norm, intrinsic_ch),
            _make_activation(activation),
        )

        # Cheap (ghost) operation — depthwise conv on intrinsic features
        self.ghost_conv = nn.Sequential(
            nn.Conv2d(
                intrinsic_ch, ghost_ch, kernel_size=(3, 3), stride=(1, 1),
                padding=(1, 1), groups=intrinsic_ch, bias=False,
            ),
            _make_norm(norm, ghost_ch),
            _make_activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B*N, C_in, F, T)`` — electrode-wise input.

        Returns
        -------
        torch.Tensor
            Shape ``(B*N, out_channels, F, T)``.
        """
        intrinsic = self.primary_conv(x)
        ghost = self.ghost_conv(intrinsic)
        return torch.cat([intrinsic, ghost], dim=1)


# ---------------------------------------------------------------------------
# DepthwiseASPP2D: multi-scale temporal receptive field without aggressive pooling
# ---------------------------------------------------------------------------

class DepthwiseASPP2D(nn.Module):
    """Depthwise Atrous Spatial Pyramid Pooling adapted for EEG.

    Dilation is primarily applied along the **time axis** (dim-3) to capture
    multi-scale temporal dynamics (delta → gamma), while keeping the frequency
    axis dilation at 1.  Each branch uses depthwise-separable convolutions for
    parameter efficiency.

    Parameters
    ----------
    in_channels : int
        Input channels.
    out_channels : int
        Output channels after fusion.
    dilations : list[int]
        Dilation rates along the time axis. Default ``[1, 2, 4, 8]``.
    norm : str
        Normalization type.
    activation : str
        Activation type.
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        dilations: list[int] | None = None,
        norm: str = "batchnorm",
        activation: str = "gelu",
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8]

        self.branches = nn.ModuleList()
        for d in dilations:
            # Depthwise conv with dilation along time axis only
            branch = nn.Sequential(
                nn.Conv2d(
                    in_channels, in_channels, kernel_size=(3, 3),
                    padding=(1, d), dilation=(1, d),
                    groups=in_channels, bias=False,
                ),
                # Pointwise projection
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                _make_norm(norm, out_channels),
                _make_activation(activation),
            )
            self.branches.append(branch)

        # 1×1 fusion after concatenation
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * len(dilations), out_channels, kernel_size=1, bias=False),
            _make_norm(norm, out_channels),
            _make_activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B*N, C, F, T)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B*N, out_channels, F, T)``.
        """
        branch_outs = [branch(x) for branch in self.branches]
        return self.fuse(torch.cat(branch_outs, dim=1))


# ---------------------------------------------------------------------------
# CoordinateAttention2D: recalibrates frequency-time features with positional info
# ---------------------------------------------------------------------------

class CoordinateAttention2D(nn.Module):
    """Coordinate Attention adapted for EEG time-frequency features.

    Encodes long-range positional information along the **frequency axis** (F)
    and the **time axis** (T) independently, then generates channel-wise
    attention maps that are position-aware.  This is more informative than
    plain Squeeze-and-Excitation for EEG because specific frequency bands
    (alpha, beta, gamma) and specific temporal windows carry distinct meaning.

    Parameters
    ----------
    in_channels : int
        Input / output channel dimension.
    reduction : int
        Channel reduction ratio in the bottleneck.  Default 4.
    norm : str
        Normalization type.
    activation : str
        Activation type.
    """

    def __init__(
        self,
        in_channels: int = 64,
        reduction: int = 4,
        norm: str = "batchnorm",
        activation: str = "gelu",
    ):
        super().__init__()
        mid = max(in_channels // reduction, 8)

        # Shared bottleneck after concatenation of F-pooled and T-pooled features
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            _make_norm(norm, mid),
            _make_activation(activation),
        )

        # Separate 1×1 conv to generate attention for F and T axes
        self.conv_f = nn.Conv2d(mid, in_channels, kernel_size=1, bias=True)
        self.conv_t = nn.Conv2d(mid, in_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B*N, C, F, T)``.

        Returns
        -------
        torch.Tensor
            Recalibrated tensor, same shape ``(B*N, C, F, T)``.
        """
        _, C, F_dim, T_dim = x.shape

        # Pool along time → frequency descriptor: (B*N, C, F, 1)
        x_f = x.mean(dim=3, keepdim=True)
        # Pool along frequency → time descriptor: (B*N, C, 1, T)
        x_t = x.mean(dim=2, keepdim=True)

        # Transpose x_t to (B*N, C, T, 1) so we can concatenate along dim-2
        x_t_perm = x_t.permute(0, 1, 3, 2)  # (B*N, C, T, 1)

        # Concatenate along spatial dim: (B*N, C, F+T, 1)
        combined = torch.cat([x_f, x_t_perm], dim=2)

        # Shared bottleneck
        combined = self.shared_conv(combined)

        # Split back into frequency and time components
        f_mid, t_mid = combined.split([F_dim, T_dim], dim=2)
        # f_mid: (B*N, mid, F, 1),  t_mid: (B*N, mid, T, 1)

        # Generate attention maps
        attn_f = self.conv_f(f_mid).sigmoid()  # (B*N, C, F, 1)
        attn_t = self.conv_t(t_mid)  # (B*N, C, T, 1)
        attn_t = attn_t.permute(0, 1, 3, 2).sigmoid()  # (B*N, C, 1, T)

        # Recalibrate: element-wise product broadcasts over missing dims
        return x * attn_f * attn_t


# ---------------------------------------------------------------------------
# NodeAttentionPool: attention-weighted pooling across electrode nodes
# ---------------------------------------------------------------------------

class NodeAttentionPool(nn.Module):
    """Attention-weighted pooling across the node (electrode) dimension.

    Produces a single vector per time step by attending over N nodes.

    Input:  (B, T, N, D)
    Output: (B, T, D)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.attn_proj = nn.Linear(d_model, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool nodes via learned attention.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, T, N, D)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, T, D)``.
        """
        # Attention weights over N nodes: (B, T, N, 1)
        scores = self.attn_proj(x)
        weights = F.softmax(scores, dim=2)
        # Weighted sum: (B, T, D)
        return (x * weights).sum(dim=2)


# ---------------------------------------------------------------------------
# EEGGhostDWASPPCASFrontEnd: combined block with ablation switches
# ---------------------------------------------------------------------------

class EEGGhostDWASPPCASFrontEnd(nn.Module):
    """Lightweight local time-frequency front-end for EEG.

    Composes GhostConv2D → DepthwiseASPP2D → CoordinateAttention2D with
    independent ablation switches.  All 2D operations are applied
    electrode-wise (``[B*N, 1, F, T]``) to preserve electrode identity
    for the downstream graph encoder.

    Parameters
    ----------
    in_freq_bins : int
        Number of frequency bins F in the input wavelet/filter-bank tensor.
    hidden_channels : int
        Internal channel dimension for Ghost / DWASPP / CAS blocks.
    out_features : int
        Output feature dimension D per electrode per time step.
    ghost_ratio : int
        Ghost module ratio. Default 2.
    dwaspp_dilations : list[int] or None
        Time-axis dilation rates for DWASPP. Default ``[1, 2, 4, 8]``.
    dropout : float
        Dropout rate before output projection.
    norm : str
        ``'batchnorm'`` or ``'layernorm'``.
    activation : str
        ``'gelu'``, ``'relu'``, or ``'silu'``.
    use_ghost : bool
        Enable Ghost convolution stem. Default True.
    use_dwaspp : bool
        Enable Depthwise ASPP. Default True.
    use_cas : bool
        Enable Coordinate Attention. Default True.
    """

    def __init__(
        self,
        in_freq_bins: int = 16,
        hidden_channels: int = 64,
        out_features: int = 64,
        ghost_ratio: int = 2,
        dwaspp_dilations: list[int] | None = None,
        dropout: float = 0.1,
        norm: str = "batchnorm",
        activation: str = "gelu",
        use_ghost: bool = True,
        use_dwaspp: bool = True,
        use_cas: bool = True,
        use_normal_conv: bool = False,
    ):
        super().__init__()
        self.use_ghost = use_ghost
        self.use_dwaspp = use_dwaspp
        self.use_cas = use_cas
        self.use_normal_conv = use_normal_conv

        # Determine the channel dimension flowing through the pipeline
        current_ch = 1  # electrode-wise, so input has 1 "image" channel

        # --- Ghost stem / Normal Conv stem ---
        if use_ghost:
            self.ghost = GhostConv2D(
                in_channels=current_ch,
                out_channels=hidden_channels,
                kernel_size=(3, 3),
                ratio=ghost_ratio,
                norm=norm,
                activation=activation,
            )
            current_ch = hidden_channels
        elif use_normal_conv:
            # Ablation: Standard 3x3 convolution to compare against Ghost
            self.ghost = nn.Sequential(
                nn.Conv2d(
                    current_ch,
                    hidden_channels,
                    kernel_size=(3, 3),
                    padding=(1, 1),
                    bias=False,
                ),
                _make_norm(norm, hidden_channels),
                _make_activation(activation),
            )
            current_ch = hidden_channels
        else:
            # Simple 1×1 projection to match hidden_channels for downstream
            self.input_proj = nn.Sequential(
                nn.Conv2d(current_ch, hidden_channels, kernel_size=1, bias=False),
                _make_norm(norm, hidden_channels),
                _make_activation(activation),
            )
            current_ch = hidden_channels

        # --- DWASPP ---
        if use_dwaspp:
            self.dwaspp = DepthwiseASPP2D(
                in_channels=current_ch,
                out_channels=hidden_channels,
                dilations=dwaspp_dilations,
                norm=norm,
                activation=activation,
            )
            current_ch = hidden_channels

        # --- CAS / Coordinate Attention ---
        if use_cas:
            self.cas = CoordinateAttention2D(
                in_channels=current_ch,
                reduction=4,
                norm=norm,
                activation=activation,
            )

        # --- Output: compress F dimension into D ---
        # After the pipeline: (B*N, hidden_channels, F, T)
        # Collapse F via 1D convolution (kernel covers full F) → (B*N, D, 1, T)
        self.freq_collapse = nn.Sequential(
            nn.Conv2d(
                hidden_channels, out_features,
                kernel_size=(in_freq_bins, 1),  # full F, keep T
                bias=False,
            ),
            _make_norm(norm, out_features),
            _make_activation(activation),
        )

        self.dropout = nn.Dropout(dropout)
        self._out_features = out_features

    @property
    def out_features(self) -> int:
        """Output feature dimension D."""
        return self._out_features

    def forward(self, x_tf: torch.Tensor) -> torch.Tensor:
        """Forward pass — electrode-wise time-frequency feature extraction.

        Parameters
        ----------
        x_tf : torch.Tensor
            Wavelet / filter-bank features, shape ``(B, N, F, T)``.

        Returns
        -------
        torch.Tensor
            Local features, shape ``(B, N, D, T)``.
            Electrode identity and temporal axis are preserved.
        """
        B, N, F_dim, T_dim = x_tf.shape

        # Reshape to electrode-wise: (B*N, 1, F, T)
        x = x_tf.reshape(B * N, 1, F_dim, T_dim)

        # Ghost or simple projection
        if self.use_ghost or self.use_normal_conv:
            x = self.ghost(x)
        else:
            x = self.input_proj(x)

        # DWASPP
        if self.use_dwaspp:
            x = self.dwaspp(x)

        # Coordinate Attention
        if self.use_cas:
            x = self.cas(x)

        # Collapse frequency: (B*N, hidden_ch, F, T) → (B*N, D, 1, T)
        x = self.freq_collapse(x)
        x = self.dropout(x)

        # Remove singleton F dim and reshape back: (B, N, D, T)
        x = x.squeeze(2)  # (B*N, D, T)
        x = x.reshape(B, N, self._out_features, T_dim)

        return x


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_norm(norm: str, channels: int) -> nn.Module:
    """Create normalization layer."""
    if norm == "batchnorm":
        return nn.BatchNorm2d(channels)
    elif norm == "layernorm":
        # GroupNorm with 1 group is equivalent to LayerNorm for conv
        return nn.GroupNorm(1, channels)
    else:
        raise ValueError(f"Unknown norm: {norm}")


def _make_activation(activation: str) -> nn.Module:
    """Create activation function."""
    if activation == "gelu":
        return nn.GELU()
    elif activation == "relu":
        return nn.ReLU(inplace=True)
    elif activation == "silu":
        return nn.SiLU(inplace=True)
    else:
        raise ValueError(f"Unknown activation: {activation}")
