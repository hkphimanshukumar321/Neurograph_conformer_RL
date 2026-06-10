"""
Shape and safety smoke tests for NeuroGraphConformer.

Verifies:
  1. Graph mode works with adj=None (auto-creates dense adjacency).
  2. Lightweight frontend + graph mode + adj=None works end-to-end.
  3. Lightweight frontend raises ValueError when frontend_type='raw'.
  4. Full end-to-end shape trace matches expected dimensions.
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from src.models.neurograph import NeuroGraphConformer


# ─── Helpers ─────────────────────────────────────────────────

def _make_cfg(**overrides) -> OmegaConf:
    """Build a minimal valid model config with optional overrides."""
    base = {
        "arch": {
            "frontend": {
                "type": "cwt",
                "freqs_min": 4,
                "freqs_max": 16,
                "time_downsample": 4,
                "projection_dim": 0,
            },
            "lightweight_frontend": {
                "enabled": False,
                "hidden_channels": 32,
                "out_features": 32,
                "ghost_ratio": 2,
                "dwaspp_dilations": [1, 2],
                "dropout": 0.0,
                "use_ghost": True,
                "use_dwaspp": True,
                "use_cas": True,
                "use_normal_conv": False,
            },
            "spatial": {
                "type": "graph",
                "n_layers": 1,
                "n_heads": 2,
                "dropout": 0.0,
                "adjacency": "distance",
            },
            "encoder": {
                "d_model": 32,
                "n_layers": 1,
                "n_heads": 2,
                "d_ff": 64,
                "conv_kernel_size": 3,
                "dropout": 0.0,
                "attention_dropout": 0.0,
                "relative_pos_encoding": False,
                "macaron": False,
            },
            "mamba": {"enabled": False},
            "heads": {
                "classification": {"dropout": 0.0},
            },
        }
    }
    cfg = OmegaConf.create(base)
    # Apply overrides using dot-notation merge
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


B, N, T = 2, 9, 512  # small batch, 9 electrodes, 512 samples (2 sec @ 250 Hz)


# ─── Test 1: Graph mode with adj=None (identity fallback) ───

class TestGraphAdjNoneFallback:
    def test_original_path_no_adj(self):
        """Original path (no lightweight frontend) should work with adj=None
        by auto-creating dense adjacency for the graph encoder."""
        cfg = _make_cfg(
            arch={"frontend": {"projection_dim": 32}}
        )
        model = NeuroGraphConformer(cfg, n_channels=N, n_samples=T, n_classes=4)
        model.eval()

        x = torch.randn(B, N, T)
        with torch.no_grad():
            out = model(x, adj=None, task="classification")

        assert "cls_logits" in out
        assert out["cls_logits"].shape == (B, 4), f"Got {out['cls_logits'].shape}"

    def test_lightweight_path_no_adj(self):
        """Lightweight path should also work with adj=None."""
        cfg = _make_cfg(
            arch={"lightweight_frontend": {"enabled": True}}
        )
        model = NeuroGraphConformer(cfg, n_channels=N, n_samples=T, n_classes=4)
        model.eval()

        x = torch.randn(B, N, T)
        with torch.no_grad():
            out = model(x, adj=None, task="classification")

        assert "cls_logits" in out
        assert out["cls_logits"].shape == (B, 4)


# ─── Test 2: Lightweight + raw frontend guard ───────────────

class TestLightweightRawGuard:
    def test_raw_frontend_raises(self):
        """Enabling lightweight frontend with raw frontend must raise ValueError."""
        cfg = _make_cfg(
            arch={
                "frontend": {"type": "raw"},
                "lightweight_frontend": {"enabled": True},
            }
        )
        with pytest.raises(ValueError, match="cwt.*filterbank"):
            NeuroGraphConformer(cfg, n_channels=N, n_samples=T, n_classes=4)

    def test_cwt_frontend_ok(self):
        """CWT frontend + lightweight should NOT raise."""
        cfg = _make_cfg(
            arch={"lightweight_frontend": {"enabled": True}}
        )
        # Should not raise
        model = NeuroGraphConformer(cfg, n_channels=N, n_samples=T, n_classes=4)
        assert model.use_lightweight_frontend is True


# ─── Test 3: Full end-to-end shape trace ─────────────────────

class TestEndToEndShapeTrace:
    def test_lightweight_shape_trace(self):
        """Verify every intermediate shape through the lightweight path."""
        cfg = _make_cfg(
            arch={"lightweight_frontend": {"enabled": True}}
        )
        model = NeuroGraphConformer(cfg, n_channels=N, n_samples=T, n_classes=4)
        model.eval()

        x = torch.randn(B, N, T)
        adj = torch.ones(N, N)  # explicit dense adjacency

        # Step 1: extract_time_frequency
        x_tf = model.frontend.extract_time_frequency(x)
        assert x_tf.dim() == 4, f"Expected 4D, got {x_tf.dim()}D"
        assert x_tf.shape[0] == B
        assert x_tf.shape[1] == N

        # Step 2: lightweight frontend
        x_local = model.lightweight_frontend(x_tf)
        assert x_local.dim() == 4
        assert x_local.shape[:2] == (B, N)

        # Step 3: full forward
        with torch.no_grad():
            out = model(x, adj=adj, task="classification")

        assert out["cls_logits"].shape == (B, 4)
        assert not torch.isnan(out["cls_logits"]).any(), "NaN in classification logits"

    def test_original_path_shape(self):
        """Original path (no lightweight frontend) should still produce valid shapes."""
        cfg = _make_cfg(
            arch={"frontend": {"projection_dim": 32}}
        )
        model = NeuroGraphConformer(cfg, n_channels=N, n_samples=T, n_classes=11)
        model.eval()

        x = torch.randn(B, N, T)
        adj = torch.ones(N, N)

        with torch.no_grad():
            out = model(x, adj=adj, task="classification")

        assert out["cls_logits"].shape == (B, 11)
        assert not torch.isnan(out["cls_logits"]).any()


# ─── Test 4: Encoder output shape ────────────────────────────

class TestEncoderOutput:
    def test_encode_lightweight(self):
        """encode() should return (B, T', d_model) in lightweight mode."""
        cfg = _make_cfg(
            arch={"lightweight_frontend": {"enabled": True}}
        )
        model = NeuroGraphConformer(cfg, n_channels=N, n_samples=T, n_classes=4)
        model.eval()

        x = torch.randn(B, N, T)
        with torch.no_grad():
            h = model.encode(x, adj=None)

        assert h.dim() == 3
        assert h.shape[0] == B
        assert h.shape[2] == 32  # d_model
        assert not torch.isnan(h).any()
