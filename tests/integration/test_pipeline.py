"""
Integration tests — End-to-end pipeline flows.

These tests verify that modules work together correctly:
  - Config → Model → Forward → Loss → Backward
  - Preprocessing → Feature extraction → Model input
  - Model → Export → Load → Verify
  - Training loop (single step)
  - Streaming inference pipeline
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import tempfile
from pathlib import Path

BATCH = 2
N_CHANNELS = 9
N_SAMPLES = 500
D_MODEL = 64
N_CLASSES = 11


class TestConfigToModel:
    """Config → Model construction → Forward pass chain."""

    def test_config_loads_and_merges(self):
        from src.utils.config import load_config, merge_configs
        base = load_config("configs/base.yaml")
        ds = load_config("configs/datasets/thinking_out_loud.yaml")
        merged = merge_configs(base, ds)
        assert merged.dataset.name == "thinking_out_loud"
        assert merged.preprocessing.target_srate == 250

    def test_model_forward_from_raw_eeg(self):
        """Full forward pass: raw EEG → all module stages → cls logits."""
        from src.models.frontend import FilterBankFrontEnd
        from src.models.graph_encoder import GraphSpatialEncoder
        from src.models.conformer import ConformerEncoder
        from src.models.heads import ClassificationHead

        eeg = torch.randn(BATCH, N_CHANNELS, N_SAMPLES)
        adj = torch.ones(N_CHANNELS, N_CHANNELS)

        # Stage 1: Frontend
        frontend = FilterBankFrontEnd(n_bands=6, sfreq=250.0, d_out=D_MODEL)
        features = frontend(eeg)
        assert features.shape == (BATCH, N_CHANNELS, D_MODEL)

        # Stage 2: Graph encoder
        graph_enc = GraphSpatialEncoder(d_in=D_MODEL, d_model=D_MODEL, n_layers=1, n_heads=4)
        spatial = graph_enc(features, adj)
        assert spatial.shape == (BATCH, N_CHANNELS, D_MODEL)

        # Stage 3: Conformer encoder
        conformer = ConformerEncoder(d_model=D_MODEL, n_layers=2, n_heads=4)
        temporal = conformer(spatial)
        assert temporal.shape == (BATCH, N_CHANNELS, D_MODEL)

        # Stage 4: Classification head
        cls_head = ClassificationHead(d_model=D_MODEL, n_classes=N_CLASSES)
        logits = cls_head(temporal)
        assert logits.shape == (BATCH, N_CLASSES)

    def test_model_forward_with_mamba(self):
        """Full forward with Mamba refinement."""
        from src.models.frontend import FilterBankFrontEnd
        from src.models.conformer import ConformerEncoder
        from src.models.mamba_module import MambaModule
        from src.models.heads import ClassificationHead

        eeg = torch.randn(BATCH, N_CHANNELS, N_SAMPLES)

        frontend = FilterBankFrontEnd(n_bands=6, sfreq=250.0, d_out=D_MODEL)
        conformer = ConformerEncoder(d_model=D_MODEL, n_layers=1, n_heads=4)
        mamba = MambaModule(d_model=D_MODEL, n_layers=1)
        head = ClassificationHead(d_model=D_MODEL, n_classes=N_CLASSES)

        x = frontend(eeg)
        x = conformer(x)
        x = mamba(x)
        logits = head(x)
        assert logits.shape == (BATCH, N_CLASSES)


class TestTrainingStep:
    """Verify a single training step (forward + loss + backward)."""

    def test_classification_train_step(self):
        from src.models.baselines import EEGNet
        from src.models.losses import FocalLoss

        model = EEGNet(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES)
        loss_fn = FocalLoss(gamma=2.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        eeg = torch.randn(BATCH, N_CHANNELS, N_SAMPLES)
        labels = torch.randint(0, N_CLASSES, (BATCH,))

        # Forward
        model.train()
        logits = model(eeg)["cls_logits"]
        loss = loss_fn(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() >= 0
        # Verify parameters were updated
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        assert grad_norm > 0

    def test_contrastive_train_step(self):
        from src.models.heads import RetrievalHead
        from src.models.conformer import ConformerEncoder
        from src.models.losses import InfoNCELoss

        encoder = ConformerEncoder(d_model=D_MODEL, n_layers=1, n_heads=4)
        head = RetrievalHead(d_model=D_MODEL, d_embed=128)
        loss_fn = InfoNCELoss(temperature=0.07)
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(head.parameters()), lr=1e-3
        )

        x = torch.randn(BATCH, N_CHANNELS, D_MODEL)
        text_emb = torch.nn.functional.normalize(torch.randn(BATCH, 128), dim=-1)

        # Forward
        encoded = encoder(x)
        eeg_emb = head(encoded)
        loss = loss_fn(eeg_emb, text_emb)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() >= 0


class TestStreamingIntegration:
    """Verify streaming inference pipeline."""

    def test_streaming_feed_and_classify(self):
        from src.models.baselines import EEGNet
        from src.deployment.streaming import StreamingInference

        model = EEGNet(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES)

        streamer = StreamingInference(
            model=model,
            window_size_s=2.0,
            stride_s=0.5,
            sfreq=250.0,
            n_channels=N_CHANNELS,
            confidence_threshold=0.0,  # Accept all
        )

        # Feed 3 seconds of data
        chunk = np.random.randn(N_CHANNELS, 750).astype(np.float32)
        results = streamer.feed(chunk)

        # Should produce predictions
        assert len(results) > 0
        assert "prediction" in results[0]
        assert "confidence" in results[0]
        assert "latency_ms" in results[0]

    def test_streaming_stats(self):
        from src.models.baselines import EEGNet
        from src.deployment.streaming import StreamingInference

        model = EEGNet(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES)
        streamer = StreamingInference(
            model=model, window_size_s=2.0, stride_s=1.0,
            sfreq=250.0, n_channels=N_CHANNELS,
        )

        # Feed multiple chunks
        for _ in range(5):
            chunk = np.random.randn(N_CHANNELS, 250).astype(np.float32)
            streamer.feed(chunk)

        stats = streamer.get_stats()
        if stats:
            assert "total_predictions" in stats
            assert "mean_latency_ms" in stats


class TestExportIntegration:
    """Verify model export and reload."""

    def test_quantize_and_infer(self):
        from src.models.baselines import EEGNet
        from src.deployment.exporter import quantize_dynamic

        model = EEGNet(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES)
        quantized = quantize_dynamic(model)

        # Should still produce valid output
        x = torch.randn(1, N_CHANNELS, N_SAMPLES)
        quantized.eval()
        with torch.no_grad():
            out = quantized(x)
        assert "cls_logits" in out
        assert out["cls_logits"].shape == (1, N_CLASSES)

    def test_profile_inference(self):
        from src.models.baselines import EEGNet
        from src.deployment.exporter import profile_inference

        model = EEGNet(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES)
        dummy = torch.randn(1, N_CHANNELS, N_SAMPLES)

        results = profile_inference(model, dummy, n_warmup=2, n_runs=5, device="cpu")
        assert "latency_mean_ms" in results
        assert "latency_p95_ms" in results
        assert "real_time_factor" in results
        assert results["n_params"] > 0


class TestCallbackIntegration:
    """Verify callbacks integrate with training loop."""

    def test_checkpoint_save_and_load(self):
        from src.training.callbacks import ModelCheckpoint
        from src.models.baselines import EEGNet

        with tempfile.TemporaryDirectory() as tmpdir:
            model = EEGNet(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            cp = ModelCheckpoint(save_dir=tmpdir, monitor="val_acc", mode="max")
            cp(model, optimizer, epoch=0, metrics={"val_acc": 0.5})

            # Check file exists
            assert (Path(tmpdir) / "best.pt").exists()

            # Load and verify
            ckpt = torch.load(Path(tmpdir) / "best.pt")
            assert "model_state_dict" in ckpt
            assert "epoch" in ckpt
            assert ckpt["metrics"]["val_acc"] == 0.5
