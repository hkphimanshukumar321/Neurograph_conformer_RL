"""Baselines package — standard EEG decoding models for comparison."""

from src.models.baselines.eegnet import EEGNet
from src.models.baselines.deepconvnet import DeepConvNet
from src.models.baselines.cnn_lstm import CNNLSTM
from src.models.baselines.cnn_bigru import CNNBiGRU
from src.models.baselines.vanilla_transformer import VanillaTransformer
from src.models.baselines.graph_only import GraphOnlyModel
from src.models.baselines.mamba_only import MambaOnlyModel

__all__ = [
    "EEGNet",
    "DeepConvNet",
    "CNNLSTM",
    "CNNBiGRU",
    "VanillaTransformer",
    "GraphOnlyModel",
    "MambaOnlyModel",
]

# Registry for config-driven model construction
BASELINE_REGISTRY = {
    "eegnet": EEGNet,
    "deepconvnet": DeepConvNet,
    "cnn_lstm": CNNLSTM,
    "cnn_bigru": CNNBiGRU,
    "vanilla_transformer": VanillaTransformer,
    "graph_only": GraphOnlyModel,
    "mamba_only": MambaOnlyModel,
}
