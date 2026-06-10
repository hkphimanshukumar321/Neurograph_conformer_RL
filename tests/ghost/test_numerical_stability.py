"""
Ghost functional tests — Numerical stability and regression detection.

Tests for:
  - NaN/Inf propagation through model chains
  - Gradient explosion/vanishing detection
  - Loss monotonicity under optimization
  - Parameter count regression (ensure model size doesn't drift)
  - Output distribution sanity checks
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

BATCH = 4
N_CHANNELS = 9
N_SAMPLES = 500
D_MODEL = 64
N_CLASSES = 11


class TestNumericalStability:
    """Detect NaN/Inf propagation in forward/backward passes."""

    def test_conformer_no_nan_backward(self):
        from src.models.conformer import ConformerEncoder
        model = ConformerEncoder(d_model=D_MODEL, n_layers=3, n_heads=4)
        x = torch.randn(BATCH, N_CHANNELS, D_MODEL, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert torch.isfinite(out).all(), "NaN in forward"
        assert torch.isfinite(x.grad).all(), "NaN in gradient"

    def test_deep_conformer_no_nan(self):
        """Deeper model should still be stable."""
        from src.models.conformer import ConformerEncoder
        model = ConformerEncoder(d_model=D_MODEL, n_layers=6, n_heads=4)
        model.eval()
        x = torch.randn(1, N_CHANNELS, D_MODEL)
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all()

    def test_graph_encoder_no_nan_asymmetric_adj(self):
        from src.models.graph_encoder import GraphSpatialEncoder
        model = GraphSpatialEncoder(d_in=D_MODEL, d_model=D_MODEL, n_layers=2, n_heads=4)
        x = torch.randn(2, N_CHANNELS, D_MODEL)
        adj = torch.rand(N_CHANNELS, N_CHANNELS)  # Asymmetric, non-binary
        out = model(x, adj)
        assert torch.isfinite(out).all()

    def test_mamba_numerical_stability(self):
        from src.models.mamba_module import MambaModule
        model = MambaModule(d_model=D_MODEL, n_layers=2)
        model.eval()
        x = torch.randn(2, 20, D_MODEL)  # Longer sequence
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all()


class TestGradientHealth:
    """Detect gradient explosion and vanishing."""

    def test_gradient_norm_bounded(self):
        """Gradient norm should be bounded for normal inputs."""
        from src.models.conformer import ConformerEncoder
        from src.models.heads import ClassificationHead

        encoder = ConformerEncoder(d_model=D_MODEL, n_layers=2, n_heads=4)
        head = ClassificationHead(d_model=D_MODEL, n_classes=N_CLASSES)

        x = torch.randn(BATCH, N_CHANNELS, D_MODEL)
        labels = torch.randint(0, N_CLASSES, (BATCH,))

        encoded = encoder(x)
        logits = head(encoded)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()

        for name, p in encoder.named_parameters():
            if p.grad is not None:
                grad_norm = p.grad.norm().item()
                assert grad_norm < 1000, f"Exploding gradient in {name}: {grad_norm}"
                # Not checking for vanishing — some params may legitimately have small grads

    def test_no_dead_parameters(self):
        """All parameters should receive gradients."""
        from src.models.conformer import ConformerEncoder
        encoder = ConformerEncoder(d_model=D_MODEL, n_layers=2, n_heads=4)
        x = torch.randn(BATCH, N_CHANNELS, D_MODEL)
        out = encoder(x)
        loss = out.sum()
        loss.backward()

        dead_params = []
        for name, p in encoder.named_parameters():
            if p.grad is None or p.grad.abs().max().item() == 0:
                dead_params.append(name)

        # Allow a small number of dead params (e.g., unused biases)
        assert len(dead_params) <= 5, f"Too many dead parameters: {dead_params}"


class TestParameterCountRegression:
    """Ensure model sizes don't drift unexpectedly."""

    def test_eegnet_param_count(self):
        from src.models.baselines import EEGNet
        model = EEGNet(n_channels=9, n_samples=500, n_classes=11)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params < 100_000, f"EEGNet too large: {n_params}"

    def test_conformer_small_param_count(self):
        from src.models.conformer import ConformerEncoder
        model = ConformerEncoder(d_model=64, n_layers=2, n_heads=4, d_ff=256)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params < 500_000, f"Conformer (small) too large: {n_params}"

    def test_baseline_param_order(self):
        """Simpler models should have fewer params than complex ones."""
        from src.models.baselines import EEGNet, DeepConvNet, VanillaTransformer

        eegnet = EEGNet(n_channels=9, n_samples=500, n_classes=11)
        deep = DeepConvNet(n_channels=9, n_samples=500, n_classes=11)
        transf = VanillaTransformer(n_channels=9, n_samples=500, n_classes=11,
                                    d_model=128, n_layers=4)

        n_eeg = sum(p.numel() for p in eegnet.parameters())
        n_deep = sum(p.numel() for p in deep.parameters())
        n_transf = sum(p.numel() for p in transf.parameters())

        # EEGNet should be smallest
        assert n_eeg < n_transf


class TestOutputDistribution:
    """Sanity check output distributions."""

    def test_softmax_sums_to_one(self):
        from src.models.baselines import EEGNet
        model = EEGNet(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES)
        model.eval()
        x = torch.randn(BATCH, N_CHANNELS, N_SAMPLES)
        with torch.no_grad():
            logits = model(x)["cls_logits"]
        probs = torch.softmax(logits, dim=-1)
        sums = probs.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(BATCH), atol=1e-5, rtol=1e-5)

    def test_retrieval_embeddings_unit_norm(self):
        from src.models.heads import RetrievalHead
        head = RetrievalHead(d_model=D_MODEL, d_embed=128)
        head.eval()
        x = torch.randn(BATCH, N_CHANNELS, D_MODEL)
        with torch.no_grad():
            emb = head(x)
        norms = torch.norm(emb, p=2, dim=-1)
        torch.testing.assert_close(norms, torch.ones(BATCH), atol=1e-4, rtol=1e-4)

    def test_logits_reasonable_range(self):
        """Logits should not be extremely large."""
        from src.models.baselines import EEGNet
        model = EEGNet(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES)
        model.eval()
        x = torch.randn(BATCH, N_CHANNELS, N_SAMPLES)
        with torch.no_grad():
            logits = model(x)["cls_logits"]
        assert logits.abs().max().item() < 100, "Logits too large"


class TestLossMonotonicity:
    """Verify loss decreases during a short training run."""

    def test_overfitting_single_batch(self):
        """Model should overfit a single batch in ~50 steps."""
        from src.models.baselines import EEGNet

        model = EEGNet(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Fixed batch
        x = torch.randn(4, N_CHANNELS, N_SAMPLES)
        y = torch.randint(0, N_CLASSES, (4,))

        losses = []
        for _ in range(50):
            model.train()
            logits = model(x)["cls_logits"]
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: start={losses[0]:.4f}, end={losses[-1]:.4f}"
        )
