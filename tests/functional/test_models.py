"""
Functional tests — Model correctness verification.

Tests gradient flow, determinism, parameter learning, loss backward,
and numerical stability of individual model components.
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


# ──── Gradient Flow ────

class TestGradientFlow:
    """Verify gradients propagate through all layers."""

    def test_conformer_gradient(self):
        from src.models.conformer import ConformerEncoder
        model = ConformerEncoder(d_model=D_MODEL, n_layers=2, n_heads=4)
        x = torch.randn(BATCH, N_CHANNELS, D_MODEL, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))

    def test_graph_encoder_gradient(self):
        from src.models.graph_encoder import GraphSpatialEncoder
        model = GraphSpatialEncoder(d_in=D_MODEL, d_model=D_MODEL, n_layers=1, n_heads=4)
        x = torch.randn(BATCH, N_CHANNELS, D_MODEL, requires_grad=True)
        adj = torch.ones(N_CHANNELS, N_CHANNELS)
        out = model(x, adj)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_mamba_gradient(self):
        from src.models.mamba_module import MambaModule
        model = MambaModule(d_model=D_MODEL, n_layers=1)
        x = torch.randn(BATCH, N_CHANNELS, D_MODEL, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_decoder_gradient(self):
        from src.models.transformer_decoder import TransformerDecoder
        model = TransformerDecoder(vocab_size=50, d_model=D_MODEL, n_layers=1, n_heads=4)
        memory = torch.randn(BATCH, 8, D_MODEL, requires_grad=True)
        tgt = torch.randint(0, 50, (BATCH, 5))
        logits = model(tgt, memory)
        loss = logits.sum()
        loss.backward()
        assert memory.grad is not None

    def test_eegnet_gradient(self):
        from src.models.baselines import EEGNet
        model = EEGNet(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES)
        x = torch.randn(BATCH, N_CHANNELS, N_SAMPLES, requires_grad=True)
        out = model(x)["cls_logits"]
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


# ──── Loss Functions ────

class TestLossFunctions:
    """Verify loss functions produce valid, differentiable outputs."""

    def test_focal_loss_backward(self):
        from src.models.losses import FocalLoss
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(BATCH, N_CLASSES, requires_grad=True)
        targets = torch.randint(0, N_CLASSES, (BATCH,))
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert loss.item() >= 0

    def test_focal_loss_reduces_on_confident(self):
        """Focal loss should be smaller for confident predictions."""
        from src.models.losses import FocalLoss
        loss_fn = FocalLoss(gamma=2.0)

        # Confident prediction
        logits_conf = torch.zeros(1, 5)
        logits_conf[0, 2] = 10.0
        loss_conf = loss_fn(logits_conf, torch.tensor([2]))

        # Uncertain prediction
        logits_unc = torch.zeros(1, 5)
        loss_unc = loss_fn(logits_unc, torch.tensor([2]))

        assert loss_conf.item() < loss_unc.item()

    def test_infonce_loss_symmetric(self):
        from src.models.losses import InfoNCELoss
        loss_fn = InfoNCELoss(temperature=0.07)
        a = torch.nn.functional.normalize(torch.randn(BATCH, 128), dim=-1)
        b = torch.nn.functional.normalize(torch.randn(BATCH, 128), dim=-1)
        loss_ab = loss_fn(a, b)
        loss_ba = loss_fn(b, a)
        # Should be symmetric (or near-symmetric)
        assert abs(loss_ab.item() - loss_ba.item()) < 0.5

    def test_ntxent_loss_positive_pairs(self):
        """NT-Xent loss should be lower for identical vs random pairs."""
        from src.models.losses import NTXentLoss
        loss_fn = NTXentLoss(temperature=0.5)
        z = torch.nn.functional.normalize(torch.randn(BATCH, 128), dim=-1)
        # Identical pairs
        loss_same = loss_fn(z, z + torch.randn_like(z) * 0.01)
        # Random pairs
        loss_rand = loss_fn(z, torch.nn.functional.normalize(torch.randn(BATCH, 128), dim=-1))
        assert loss_same.item() < loss_rand.item()

    def test_scst_loss_shape(self):
        from src.models.losses import SCSTLoss
        loss_fn = SCSTLoss()
        log_probs = torch.randn(BATCH, 5)
        sample_r = torch.randn(BATCH)
        greedy_r = torch.randn(BATCH)
        loss = loss_fn(log_probs, sample_r, greedy_r)
        assert loss.dim() == 0  # Scalar
        assert torch.isfinite(loss)

    def test_ctc_loss_backward(self):
        from src.models.losses import CTCLoss
        loss_fn = CTCLoss(blank_id=0)
        logits = torch.randn(20, BATCH, 30, requires_grad=True)
        log_probs = logits.log_softmax(2)
        targets = torch.randint(1, 30, (BATCH, 5))
        input_lengths = torch.full((BATCH,), 20, dtype=torch.long)
        target_lengths = torch.full((BATCH,), 5, dtype=torch.long)
        loss = loss_fn(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        assert logits.grad is not None


# ──── Determinism ────

class TestDeterminism:
    """Verify models produce deterministic outputs with same seed."""

    def test_conformer_deterministic(self):
        from src.models.conformer import ConformerEncoder
        from src.utils.seed import seed_everything

        model = ConformerEncoder(d_model=D_MODEL, n_layers=1, n_heads=4)
        model.eval()

        x = torch.randn(1, N_CHANNELS, D_MODEL)

        seed_everything(42)
        out1 = model(x)

        seed_everything(42)
        out2 = model(x)

        torch.testing.assert_close(out1, out2)

    def test_baseline_deterministic(self):
        from src.models.baselines import EEGNet
        from src.utils.seed import seed_everything

        seed_everything(42)
        model = EEGNet(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES)
        model.eval()
        x = torch.randn(1, N_CHANNELS, N_SAMPLES)

        out1 = model(x)["cls_logits"]
        out2 = model(x)["cls_logits"]
        torch.testing.assert_close(out1, out2)


# ──── Retrieval Head ────

class TestRetrievalHead:
    """Verify retrieval head produces normalized embeddings."""

    def test_l2_normalization(self):
        from src.models.heads import RetrievalHead
        head = RetrievalHead(d_model=D_MODEL, d_embed=128)
        x = torch.randn(BATCH, N_CHANNELS, D_MODEL)
        emb = head(x)
        norms = torch.norm(emb, p=2, dim=-1)
        torch.testing.assert_close(norms, torch.ones(BATCH), atol=1e-5, rtol=1e-5)

    def test_similarity_range(self):
        from src.models.heads import RetrievalHead
        head = RetrievalHead(d_model=D_MODEL, d_embed=128)
        x = torch.randn(BATCH, N_CHANNELS, D_MODEL)
        emb = head(x)
        sim = torch.mm(emb, emb.t())
        # All cosine similarities should be in [-1, 1]
        assert torch.all(sim >= -1.01)
        assert torch.all(sim <= 1.01)
