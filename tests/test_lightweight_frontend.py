"""
Smoke tests for the lightweight EEG front-end modules.

Verifies:
  - Correct output shapes through each sub-module and the combined front-end.
  - No NaN outputs for random inputs.
  - All ablation switch combinations (Ghost / DWASPP / CAS) work independently.
  - NodeAttentionPool produces correct shapes.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from src.models.eeg_lightweight_frontend import (
    CoordinateAttention2D,
    DepthwiseASPP2D,
    EEGGhostDWASPPCASFrontEnd,
    GhostConv2D,
    NodeAttentionPool,
)

# Standard test dimensions
B = 2       # batch
N = 64      # EEG electrodes / graph nodes
F = 16      # frequency bins (wavelet scales)
T = 128     # time frames
H = 64      # hidden channels
D_OUT = 64  # output feature dim


# ──────────────────────────────────────────────────────────────
# Individual module tests
# ──────────────────────────────────────────────────────────────

class TestGhostConv2D:
    def test_shape(self):
        """GhostConv2D should output (B*N, out_ch, F, T)."""
        m = GhostConv2D(in_channels=1, out_channels=H, ratio=2)
        x = torch.randn(B * N, 1, F, T)
        out = m(x)
        assert out.shape == (B * N, H, F, T), f"Expected {(B * N, H, F, T)}, got {out.shape}"

    def test_no_nan(self):
        m = GhostConv2D(in_channels=1, out_channels=H, ratio=2)
        x = torch.randn(B * N, 1, F, T)
        out = m(x)
        assert not torch.isnan(out).any(), "NaN detected in GhostConv2D output"


class TestDepthwiseASPP2D:
    def test_shape(self):
        """DepthwiseASPP2D should preserve spatial dims."""
        m = DepthwiseASPP2D(in_channels=H, out_channels=H, dilations=[1, 2, 4, 8])
        x = torch.randn(B * N, H, F, T)
        out = m(x)
        assert out.shape == (B * N, H, F, T), f"Expected {(B * N, H, F, T)}, got {out.shape}"

    def test_no_nan(self):
        m = DepthwiseASPP2D(in_channels=H, out_channels=H)
        x = torch.randn(B * N, H, F, T)
        out = m(x)
        assert not torch.isnan(out).any(), "NaN detected in DepthwiseASPP2D output"


class TestCoordinateAttention2D:
    def test_shape(self):
        """CoordinateAttention2D should preserve all dims."""
        m = CoordinateAttention2D(in_channels=H, reduction=4)
        x = torch.randn(B * N, H, F, T)
        out = m(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_no_nan(self):
        m = CoordinateAttention2D(in_channels=H)
        x = torch.randn(B * N, H, F, T)
        out = m(x)
        assert not torch.isnan(out).any(), "NaN detected in CoordinateAttention2D output"


class TestNodeAttentionPool:
    def test_shape(self):
        """NodeAttentionPool: (B, T, N, D) → (B, T, D)."""
        m = NodeAttentionPool(d_model=D_OUT)
        x = torch.randn(B, T, N, D_OUT)
        out = m(x)
        assert out.shape == (B, T, D_OUT), f"Expected {(B, T, D_OUT)}, got {out.shape}"

    def test_no_nan(self):
        m = NodeAttentionPool(d_model=D_OUT)
        x = torch.randn(B, T, N, D_OUT)
        out = m(x)
        assert not torch.isnan(out).any(), "NaN detected in NodeAttentionPool output"


# ──────────────────────────────────────────────────────────────
# Combined front-end tests
# ──────────────────────────────────────────────────────────────

class TestEEGGhostDWASPPCASFrontEnd:
    def test_full_frontend_shape(self):
        """Combined front-end: (B, N, F, T) → (B, N, D, T)."""
        m = EEGGhostDWASPPCASFrontEnd(
            in_freq_bins=F,
            hidden_channels=H,
            out_features=D_OUT,
            use_ghost=True,
            use_dwaspp=True,
            use_cas=True,
        )
        x = torch.randn(B, N, F, T)
        out = m(x)
        assert out.shape == (B, N, D_OUT, T), f"Expected {(B, N, D_OUT, T)}, got {out.shape}"

    def test_full_frontend_no_nan(self):
        m = EEGGhostDWASPPCASFrontEnd(
            in_freq_bins=F, hidden_channels=H, out_features=D_OUT,
        )
        x = torch.randn(B, N, F, T)
        out = m(x)
        assert not torch.isnan(out).any(), "NaN detected in combined front-end output"

    @pytest.mark.parametrize(
        "use_ghost,use_dwaspp,use_cas",
        list(itertools.product([True, False], repeat=3)),
        ids=lambda combo: f"g{int(combo)}" if isinstance(combo, bool) else None,
    )
    def test_ablation_variants(self, use_ghost: bool, use_dwaspp: bool, use_cas: bool):
        """Every ablation switch combination must produce correct shapes and no NaN."""
        m = EEGGhostDWASPPCASFrontEnd(
            in_freq_bins=F,
            hidden_channels=H,
            out_features=D_OUT,
            use_ghost=use_ghost,
            use_dwaspp=use_dwaspp,
            use_cas=use_cas,
        )
        x = torch.randn(B, N, F, T)
        out = m(x)
        assert out.shape == (B, N, D_OUT, T), (
            f"Ablation (ghost={use_ghost}, dwaspp={use_dwaspp}, cas={use_cas}): "
            f"expected {(B, N, D_OUT, T)}, got {out.shape}"
        )
        assert not torch.isnan(out).any(), (
            f"NaN for ablation (ghost={use_ghost}, dwaspp={use_dwaspp}, cas={use_cas})"
        )


# ──────────────────────────────────────────────────────────────
# End-to-end shape trace (prints for manual review)
# ──────────────────────────────────────────────────────────────

class TestEndToEndShapeTrace:
    def test_print_shape_trace(self, capsys):
        """Print the full shape trace through the lightweight front-end pipeline."""
        frontend = EEGGhostDWASPPCASFrontEnd(
            in_freq_bins=F, hidden_channels=H, out_features=D_OUT,
        )
        node_pool = NodeAttentionPool(d_model=D_OUT)

        x_tf = torch.randn(B, N, F, T)
        print(f"Input  x_tf:         {list(x_tf.shape)}  [B, N, F, T]")

        x_local = frontend(x_tf)
        print(f"After frontend:      {list(x_local.shape)}  [B, N, D, T]")

        D = x_local.shape[2]
        T_dim = x_local.shape[3]

        # Rearrange for graph encoder
        x_graph_in = x_local.permute(0, 3, 1, 2).contiguous().reshape(B * T_dim, N, D)
        print(f"Graph input:         {list(x_graph_in.shape)}  [(B*T), N, D]")

        # Simulate graph encoder (identity for shape check)
        x_graph_out = x_graph_in  # pretend D_g == D
        x_graph_out = x_graph_out.reshape(B, T_dim, N, D)
        print(f"Graph output:        {list(x_graph_out.shape)}  [B, T, N, D_g]")

        x_seq = node_pool(x_graph_out)
        print(f"After node pool:     {list(x_seq.shape)}  [B, T, D_g]")

        # Assert final shape is compatible with Conformer input
        assert x_seq.shape == (B, T_dim, D), f"Expected (B, T, D_g), got {x_seq.shape}"

        captured = capsys.readouterr()
        assert "Input  x_tf" in captured.out
