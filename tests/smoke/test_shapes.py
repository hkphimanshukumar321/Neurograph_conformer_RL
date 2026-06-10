"""
Smoke tests — Forward pass shape verification.

Ensures all models produce correct output tensor shapes.
Catches dimension mismatches, transpose errors, pooling bugs.
"""

import pytest
import torch

BATCH = 2
N_CHANNELS = 9
N_SAMPLES = 500
D_MODEL = 64
N_CLASSES = 11


@pytest.fixture
def eeg():
    return torch.randn(BATCH, N_CHANNELS, N_SAMPLES)


@pytest.fixture
def seq():
    return torch.randn(BATCH, N_CHANNELS, D_MODEL)


@pytest.fixture
def adj():
    return torch.ones(N_CHANNELS, N_CHANNELS)


class TestForwardShapes:
    """Verify output shapes of all core modules."""

    def test_filterbank_shape(self, eeg):
        from src.models.frontend import FilterBankFrontEnd
        m = FilterBankFrontEnd(n_bands=6, sfreq=250.0, d_out=D_MODEL)
        out = m(eeg)
        assert out.shape == (BATCH, N_CHANNELS, D_MODEL)

    def test_graph_encoder_shape(self, seq, adj):
        from src.models.graph_encoder import GraphSpatialEncoder
        m = GraphSpatialEncoder(d_in=D_MODEL, d_model=D_MODEL, n_layers=1, n_heads=4)
        out = m(seq, adj)
        assert out.shape == (BATCH, N_CHANNELS, D_MODEL)

    def test_conformer_shape(self, seq):
        from src.models.conformer import ConformerEncoder
        m = ConformerEncoder(d_model=D_MODEL, n_layers=1, n_heads=4, d_ff=128)
        out = m(seq)
        assert out.shape == seq.shape

    def test_mamba_shape(self, seq):
        from src.models.mamba_module import MambaModule
        m = MambaModule(d_model=D_MODEL, n_layers=1)
        out = m(seq)
        assert out.shape == seq.shape

    def test_classification_head_shape(self, seq):
        from src.models.heads import ClassificationHead
        m = ClassificationHead(d_model=D_MODEL, n_classes=N_CLASSES)
        out = m(seq)
        assert out.shape == (BATCH, N_CLASSES)

    def test_retrieval_head_shape(self, seq):
        from src.models.heads import RetrievalHead
        m = RetrievalHead(d_model=D_MODEL, d_embed=128)
        out = m(seq)
        assert out.shape == (BATCH, 128)

    def test_decoder_teacher_forcing_shape(self):
        from src.models.transformer_decoder import TransformerDecoder
        m = TransformerDecoder(vocab_size=50, d_model=D_MODEL, n_layers=1, n_heads=4)
        memory = torch.randn(BATCH, 10, D_MODEL)
        tgt = torch.randint(0, 50, (BATCH, 5))
        logits = m(tgt, memory)
        assert logits.shape == (BATCH, 5, 50)


class TestBaselineForwardShapes:
    """Verify all 7 baselines produce correct cls_logits shape."""

    @pytest.mark.parametrize("model_name", [
        "eegnet", "deepconvnet", "cnn_lstm", "cnn_bigru",
        "vanilla_transformer", "mamba_only",
    ])
    def test_baseline_cls_logits(self, eeg, model_name):
        from src.models.baselines import BASELINE_REGISTRY

        kwargs = {"n_channels": N_CHANNELS, "n_samples": N_SAMPLES, "n_classes": N_CLASSES}
        if model_name in ("vanilla_transformer", "mamba_only"):
            kwargs["d_model"] = D_MODEL
            kwargs["n_layers"] = 1

        model_cls = BASELINE_REGISTRY[model_name]
        model = model_cls(**kwargs)
        out = model(eeg)

        assert "cls_logits" in out
        assert out["cls_logits"].shape == (BATCH, N_CLASSES)
