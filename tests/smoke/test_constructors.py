"""
Smoke tests — Constructor verification.

Ensures all models and modules can be instantiated with default parameters.
These tests catch parameter mismatches, missing defaults, and constructor bugs.
"""

import pytest
import torch


BATCH = 2
N_CHANNELS = 9
N_SAMPLES = 500
D_MODEL = 64
N_CLASSES = 11


class TestConstructModels:
    """Verify all models construct without error."""

    def test_construct_wavelet_frontend(self):
        from src.models.frontend import WaveletFrontEnd
        m = WaveletFrontEnd(d_out=D_MODEL)
        assert m is not None

    def test_construct_filterbank_frontend(self):
        from src.models.frontend import FilterBankFrontEnd
        m = FilterBankFrontEnd(n_bands=6, d_out=D_MODEL)
        assert m is not None

    def test_construct_graph_encoder(self):
        from src.models.graph_encoder import GraphSpatialEncoder
        m = GraphSpatialEncoder(d_in=D_MODEL, d_model=D_MODEL)
        assert m is not None

    def test_construct_conformer(self):
        from src.models.conformer import ConformerEncoder
        m = ConformerEncoder(d_model=D_MODEL, n_layers=1, n_heads=4)
        assert m is not None

    def test_construct_mamba(self):
        from src.models.mamba_module import MambaModule
        m = MambaModule(d_model=D_MODEL, n_layers=1)
        assert m is not None

    def test_construct_decoder(self):
        from src.models.transformer_decoder import TransformerDecoder
        m = TransformerDecoder(vocab_size=100, d_model=D_MODEL, n_layers=1, n_heads=4)
        assert m is not None

    def test_construct_classification_head(self):
        from src.models.heads import ClassificationHead
        m = ClassificationHead(d_model=D_MODEL, n_classes=N_CLASSES)
        assert m is not None

    def test_construct_retrieval_head(self):
        from src.models.heads import RetrievalHead
        m = RetrievalHead(d_model=D_MODEL, d_embed=128)
        assert m is not None


class TestConstructBaselines:
    """Verify all 7 baselines construct."""

    def test_construct_eegnet(self):
        from src.models.baselines import EEGNet
        m = EEGNet(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES)
        assert sum(p.numel() for p in m.parameters()) > 0

    def test_construct_deepconvnet(self):
        from src.models.baselines import DeepConvNet
        m = DeepConvNet(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES)
        assert sum(p.numel() for p in m.parameters()) > 0

    def test_construct_cnn_lstm(self):
        from src.models.baselines import CNNLSTM
        m = CNNLSTM(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES)
        assert sum(p.numel() for p in m.parameters()) > 0

    def test_construct_cnn_bigru(self):
        from src.models.baselines import CNNBiGRU
        m = CNNBiGRU(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES)
        assert sum(p.numel() for p in m.parameters()) > 0

    def test_construct_vanilla_transformer(self):
        from src.models.baselines import VanillaTransformer
        m = VanillaTransformer(n_channels=N_CHANNELS, n_samples=N_SAMPLES,
                               n_classes=N_CLASSES, d_model=D_MODEL, n_layers=1)
        assert sum(p.numel() for p in m.parameters()) > 0

    def test_construct_graph_only(self):
        from src.models.baselines import GraphOnlyModel
        m = GraphOnlyModel(n_channels=N_CHANNELS, n_samples=N_SAMPLES,
                           n_classes=N_CLASSES, d_model=D_MODEL)
        assert sum(p.numel() for p in m.parameters()) > 0

    def test_construct_mamba_only(self):
        from src.models.baselines import MambaOnlyModel
        m = MambaOnlyModel(n_channels=N_CHANNELS, n_samples=N_SAMPLES,
                           n_classes=N_CLASSES, d_model=D_MODEL, n_layers=1)
        assert sum(p.numel() for p in m.parameters()) > 0


class TestConstructTraining:
    """Verify training components construct."""

    def test_construct_early_stopping(self):
        from src.training.callbacks import EarlyStopping
        cb = EarlyStopping(patience=10, monitor="val_loss", mode="min")
        assert not cb.should_stop

    def test_construct_augmentation_compose(self):
        from src.training.augmentation import (
            Compose, TimeShift, GaussianNoise, ChannelDropout,
        )
        aug = Compose([TimeShift(25), GaussianNoise(0.1), ChannelDropout(0.1)])
        assert len(aug.transforms) == 3

    def test_construct_scheduler(self):
        from src.training.schedulers import build_scheduler
        opt = torch.optim.Adam([torch.randn(5, requires_grad=True)], lr=1e-3)
        sched = build_scheduler(opt, "cosine_warmup", warmup_epochs=5, max_epochs=50)
        assert sched is not None

    def test_construct_scheduler_none(self):
        from src.training.schedulers import build_scheduler
        opt = torch.optim.Adam([torch.randn(5, requires_grad=True)], lr=1e-3)
        sched = build_scheduler(opt, "none")
        assert sched is None


class TestConstructConfig:
    """Verify config loading constructs properly."""

    def test_load_base_config(self):
        from src.utils.config import load_config
        cfg = load_config("configs/base.yaml")
        assert cfg is not None
        assert "preprocessing" in cfg

    def test_load_dataset_configs(self):
        from src.utils.config import load_config
        for name in ["kara_one", "thinking_out_loud", "chisco", "zuco"]:
            cfg = load_config(f"configs/datasets/{name}.yaml")
            assert cfg.dataset.name == name
