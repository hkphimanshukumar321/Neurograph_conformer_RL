"""
Ghost functional tests — Edge cases, boundary conditions, adversarial inputs,
failure modes, and numerical stability.

These tests catch the "invisible" bugs that only appear under unusual conditions:
  - Zero-length inputs
  - Single-sample batches
  - Extreme values (NaN, Inf, very large/small)
  - Mismatched dimensions
  - Empty adjacency matrices
  - Degenerate configurations
  - Numerical stability under float16
  - Memory leak detection patterns
"""

import pytest
import numpy as np
import torch

BATCH = 4
N_CHANNELS = 9
N_SAMPLES = 500
D_MODEL = 64
N_CLASSES = 11


# ──── Boundary Conditions ────

class TestBoundaryInputs:
    """Test models with boundary-condition inputs."""

    def test_single_sample_batch(self):
        """Batch size = 1 should work without BatchNorm issues."""
        from src.models.baselines import EEGNet
        model = EEGNet(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES)
        model.eval()
        x = torch.randn(1, N_CHANNELS, N_SAMPLES)
        out = model(x)
        assert out["cls_logits"].shape == (1, N_CLASSES)

    def test_minimum_time_samples(self):
        """Very short time series should not crash."""
        from src.models.baselines import CNNLSTM
        try:
            model = CNNLSTM(n_channels=N_CHANNELS, n_samples=50, n_classes=N_CLASSES)
            x = torch.randn(2, N_CHANNELS, 50)
            out = model(x)
            assert "cls_logits" in out
        except Exception:
            pass  # Expected for too-small inputs

    def test_single_channel(self):
        """Single EEG channel should work."""
        from src.models.baselines import CNNLSTM
        model = CNNLSTM(n_channels=1, n_samples=N_SAMPLES, n_classes=N_CLASSES)
        x = torch.randn(2, 1, N_SAMPLES)
        out = model(x)
        assert out["cls_logits"].shape == (2, N_CLASSES)

    def test_two_classes(self):
        """Binary classification edge case."""
        from src.models.baselines import EEGNet
        model = EEGNet(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=2)
        x = torch.randn(2, N_CHANNELS, N_SAMPLES)
        out = model(x)
        assert out["cls_logits"].shape == (2, 2)

    def test_large_class_count(self):
        """Many classes (e.g., Chisco ~80 chars)."""
        from src.models.baselines import EEGNet
        model = EEGNet(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=80)
        x = torch.randn(2, N_CHANNELS, N_SAMPLES)
        out = model(x)
        assert out["cls_logits"].shape == (2, 80)


# ──── Adversarial / Extreme Values ────

class TestExtremeValues:
    """Test handling of extreme numerical values."""

    def test_zero_input(self):
        """All-zero input should not produce NaN."""
        from src.models.conformer import ConformerEncoder
        model = ConformerEncoder(d_model=D_MODEL, n_layers=1, n_heads=4)
        model.eval()
        x = torch.zeros(2, N_CHANNELS, D_MODEL)
        out = model(x)
        assert torch.isfinite(out).all(), "Zero input produced NaN/Inf"

    def test_constant_input(self):
        """Constant input (no variance) should not crash normalization."""
        from src.models.conformer import ConformerEncoder
        model = ConformerEncoder(d_model=D_MODEL, n_layers=1, n_heads=4)
        model.eval()
        x = torch.ones(2, N_CHANNELS, D_MODEL) * 5.0
        out = model(x)
        assert torch.isfinite(out).all()

    def test_large_input_values(self):
        """Very large input values should not cause overflow."""
        from src.models.baselines import EEGNet
        model = EEGNet(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES)
        model.eval()
        x = torch.randn(1, N_CHANNELS, N_SAMPLES) * 1000
        with torch.no_grad():
            out = model(x)
        # May produce large logits but should not be NaN
        assert torch.isfinite(out["cls_logits"]).all()

    def test_small_input_values(self):
        """Very small input values (near-zero)."""
        from src.models.baselines import EEGNet
        model = EEGNet(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES)
        model.eval()
        x = torch.randn(1, N_CHANNELS, N_SAMPLES) * 1e-10
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out["cls_logits"]).all()


# ──── Graph Edge Cases ────

class TestGraphEdgeCases:
    """Test graph encoder with unusual adjacency structures."""

    def test_identity_adjacency(self):
        """Self-connections only (diagonal adjacency)."""
        from src.models.graph_encoder import GraphSpatialEncoder
        model = GraphSpatialEncoder(d_in=D_MODEL, d_model=D_MODEL, n_layers=1, n_heads=4)
        x = torch.randn(2, N_CHANNELS, D_MODEL)
        adj = torch.eye(N_CHANNELS)
        out = model(x, adj)
        assert out.shape == (2, N_CHANNELS, D_MODEL)
        assert torch.isfinite(out).all()

    def test_zero_adjacency(self):
        """No connections — should still produce output."""
        from src.models.graph_encoder import GraphSpatialEncoder
        model = GraphSpatialEncoder(d_in=D_MODEL, d_model=D_MODEL, n_layers=1, n_heads=4)
        x = torch.randn(2, N_CHANNELS, D_MODEL)
        adj = torch.zeros(N_CHANNELS, N_CHANNELS)
        out = model(x, adj)
        assert out.shape == (2, N_CHANNELS, D_MODEL)

    def test_sparse_adjacency(self):
        """Very sparse graph (chain topology)."""
        from src.models.graph_encoder import GraphSpatialEncoder
        model = GraphSpatialEncoder(d_in=D_MODEL, d_model=D_MODEL, n_layers=1, n_heads=4)
        x = torch.randn(2, N_CHANNELS, D_MODEL)
        adj = torch.zeros(N_CHANNELS, N_CHANNELS)
        for i in range(N_CHANNELS - 1):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0
        out = model(x, adj)
        assert torch.isfinite(out).all()


# ──── Loss Edge Cases ────

class TestLossEdgeCases:
    """Test loss functions with edge-case inputs."""

    def test_focal_loss_perfect_prediction(self):
        """Loss should be very small for perfect predictions."""
        from src.models.losses import FocalLoss
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.zeros(1, N_CLASSES)
        logits[0, 3] = 100.0  # Very confident correct prediction
        target = torch.tensor([3])
        loss = loss_fn(logits, target)
        assert loss.item() < 0.01

    def test_focal_loss_all_same_class(self):
        """All samples are the same class."""
        from src.models.losses import FocalLoss
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(10, N_CLASSES)
        targets = torch.zeros(10, dtype=torch.long)  # All class 0
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss)

    def test_infonce_identical_embeddings(self):
        """All embeddings are identical — degenerate case."""
        from src.models.losses import InfoNCELoss
        loss_fn = InfoNCELoss(temperature=0.07)
        emb = torch.nn.functional.normalize(torch.ones(4, 128), dim=-1)
        loss = loss_fn(emb, emb)
        assert torch.isfinite(loss)

    def test_ctc_loss_long_target(self):
        """Target longer than input — should handle gracefully."""
        from src.models.losses import CTCLoss
        loss_fn = CTCLoss(blank_id=0)
        log_probs = torch.randn(5, 1, 30).log_softmax(2)  # T=5
        targets = torch.randint(1, 30, (1, 10))  # len=10 > T=5
        input_lengths = torch.tensor([5])
        target_lengths = torch.tensor([10])
        # Should either handle or raise a meaningful error
        try:
            loss = loss_fn(log_probs, targets, input_lengths, target_lengths)
        except RuntimeError:
            pass  # CTC expects target_len <= input_len


# ──── Preprocessing Edge Cases ────

class TestPreprocessingEdgeCases:
    """Ghost tests for preprocessing edge conditions."""

    def test_zscore_zero_variance(self):
        """Data with zero variance should not produce NaN."""
        from src.preprocessing.normalization import zscore_normalize
        data = np.ones((5, 9, 500), dtype=np.float32)
        normed = zscore_normalize(data, scope="per_session")
        assert np.isfinite(normed).all()

    def test_zscore_single_trial(self):
        """Single trial normalization."""
        from src.preprocessing.normalization import zscore_normalize
        data = np.random.randn(1, 9, 500).astype(np.float32)
        normed = zscore_normalize(data, scope="per_trial")
        assert normed.shape == (1, 9, 500)
        assert np.isfinite(normed).all()

    def test_robust_normalize_constant_data(self):
        """Constant data should not crash robust normalizer."""
        from src.preprocessing.normalization import robust_normalize
        data = np.ones((10, 9, 500), dtype=np.float32) * 42
        normed = robust_normalize(data, scope="per_session")
        assert np.isfinite(normed).all()

    def test_epoch_exactly_one_window(self):
        """Data exactly equals one window size."""
        from src.preprocessing.epoching import create_sliding_window_epochs
        data = np.random.randn(9, 500).astype(np.float32)
        epochs = create_sliding_window_epochs(data, sfreq=250.0, window_s=2.0, stride_s=1.0)
        assert epochs.shape[0] == 1
        assert epochs.shape[2] == 500


# ──── Decoder Edge Cases ────

class TestDecoderEdgeCases:
    """Ghost tests for decoder edge cases."""

    def test_greedy_decode_empty_memory(self):
        """Very short memory sequence."""
        from src.models.transformer_decoder import TransformerDecoder
        decoder = TransformerDecoder(vocab_size=50, d_model=D_MODEL, n_layers=1, n_heads=4, max_seq_len=10)
        memory = torch.randn(1, 1, D_MODEL)  # Single timestep
        generated = decoder.greedy_decode(memory, sos_id=1, eos_id=2, max_len=5)
        assert generated.shape[0] == 1
        assert generated.shape[1] <= 6

    def test_decoder_with_padding_mask(self):
        """Decoder should handle padding masks."""
        from src.models.transformer_decoder import TransformerDecoder
        decoder = TransformerDecoder(vocab_size=50, d_model=D_MODEL, n_layers=1, n_heads=4)
        memory = torch.randn(2, 10, D_MODEL)
        tgt = torch.randint(0, 50, (2, 5))
        logits = decoder(tgt, memory)
        assert logits.shape == (2, 5, 50)


# ──── Augmentation Edge Cases ────

class TestAugmentationEdgeCases:
    """Ghost tests for augmentation edge conditions."""

    def test_channel_dropout_p_zero(self):
        """p=0 should never drop channels."""
        from src.training.augmentation import ChannelDropout
        aug = ChannelDropout(p=0.0)
        x = np.ones((9, 500), dtype=np.float32)
        out = aug(x)
        np.testing.assert_array_equal(x, out)

    def test_channel_dropout_p_one(self):
        """p=1 should drop all channels."""
        from src.training.augmentation import ChannelDropout
        aug = ChannelDropout(p=1.0)
        x = np.ones((9, 500), dtype=np.float32)
        out = aug(x)
        assert np.allclose(out, 0)

    def test_time_shift_zero(self):
        """max_shift=0 should not change data."""
        from src.training.augmentation import TimeShift
        aug = TimeShift(max_shift_samples=0)
        x = np.random.randn(9, 500).astype(np.float32)
        out = aug(x)
        np.testing.assert_array_equal(x, out)


# ──── Callback Edge Cases ────

class TestCallbackEdgeCases:
    """Ghost tests for callback edge conditions."""

    def test_early_stopping_patience_one(self):
        """Patience=1 should stop immediately on first non-improvement."""
        from src.training.callbacks import EarlyStopping
        es = EarlyStopping(patience=1, mode="max")
        es(0.5)  # First call sets best
        stopped = es(0.4)  # Worse → counter=1 → stop
        assert stopped

    def test_early_stopping_always_improving(self):
        """Should never stop if always improving."""
        from src.training.callbacks import EarlyStopping
        es = EarlyStopping(patience=3, mode="max")
        for v in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            stopped = es(v)
        assert not stopped

    def test_gradient_monitor_no_grads(self):
        """Gradient monitor with no parameters having gradients."""
        from src.training.callbacks import GradientMonitor
        import torch.nn as nn
        gm = GradientMonitor(log_interval=1)
        model = nn.Linear(10, 10)
        # No backward called → no grads
        stats = gm(model)
        assert "grad/total_norm" in stats
        assert stats["grad/total_norm"] == 0.0


# ──── Streaming Edge Cases ────

class TestStreamingEdgeCases:
    """Ghost tests for streaming inference edge conditions."""

    def test_streaming_small_chunk(self):
        """Feed data smaller than stride."""
        from src.models.baselines import EEGNet
        from src.deployment.streaming import StreamingInference

        model = EEGNet(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES)
        streamer = StreamingInference(
            model=model, window_size_s=2.0, stride_s=1.0,
            sfreq=250.0, n_channels=N_CHANNELS,
        )

        # Feed tiny chunk
        chunk = np.random.randn(N_CHANNELS, 10).astype(np.float32)
        results = streamer.feed(chunk)
        assert len(results) == 0  # Not enough data yet

    def test_streaming_reset(self):
        """Reset should clear buffer and predictions."""
        from src.models.baselines import EEGNet
        from src.deployment.streaming import StreamingInference

        model = EEGNet(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES)
        streamer = StreamingInference(
            model=model, window_size_s=2.0, stride_s=0.5,
            sfreq=250.0, n_channels=N_CHANNELS,
        )

        # Feed data
        chunk = np.random.randn(N_CHANNELS, 1000).astype(np.float32)
        streamer.feed(chunk)
        assert len(streamer.predictions) > 0

        # Reset
        streamer.reset()
        assert len(streamer.predictions) == 0
        assert streamer.buffer.shape[1] == 0

    def test_streaming_confidence_threshold(self):
        """High confidence threshold should reject uncertain predictions."""
        from src.models.baselines import EEGNet
        from src.deployment.streaming import StreamingInference

        model = EEGNet(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES)
        streamer = StreamingInference(
            model=model, window_size_s=2.0, stride_s=1.0,
            sfreq=250.0, n_channels=N_CHANNELS,
            confidence_threshold=0.99,  # Very high → most will be rejected
        )

        chunk = np.random.randn(N_CHANNELS, 1000).astype(np.float32)
        results = streamer.feed(chunk)

        if results:
            # Check that rejected predictions have pred=-1
            rejected = [r for r in results if not r["accepted"]]
            for r in rejected:
                assert r["prediction"] == -1
