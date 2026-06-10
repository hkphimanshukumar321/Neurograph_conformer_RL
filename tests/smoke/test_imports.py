"""
Smoke tests — Import verification.

Ensures all modules can be imported without errors.
These run in <2 seconds and catch broken imports, missing deps, syntax errors.
"""

import pytest


class TestImportsModels:
    """Verify all model modules import without error."""

    def test_import_frontend(self):
        from src.models.frontend import WaveletFrontEnd, FilterBankFrontEnd

    def test_import_graph_encoder(self):
        from src.models.graph_encoder import GraphSpatialEncoder, GATv2Layer

    def test_import_conformer(self):
        from src.models.conformer import ConformerEncoder, ConformerBlock

    def test_import_mamba(self):
        from src.models.mamba_module import MambaModule

    def test_import_decoder(self):
        from src.models.transformer_decoder import TransformerDecoder

    def test_import_heads(self):
        from src.models.heads import ClassificationHead, RetrievalHead, GenerationHead

    def test_import_losses(self):
        from src.models.losses import (
            FocalLoss, InfoNCELoss, NTXentLoss, TripletLoss,
            CTCLoss, SCSTLoss, MWERLoss,
        )

    def test_import_neurograph(self):
        from src.models.neurograph import NeuroGraphConformer

    def test_import_baselines(self):
        from src.models.baselines import (
            EEGNet, DeepConvNet, CNNLSTM, CNNBiGRU,
            VanillaTransformer, GraphOnlyModel, MambaOnlyModel,
        )

    def test_import_baseline_registry(self):
        from src.models.baselines import BASELINE_REGISTRY
        assert len(BASELINE_REGISTRY) == 7


class TestImportsPreprocessing:
    """Verify all preprocessing modules import."""

    def test_import_pipeline(self):
        from src.preprocessing.pipeline import PreprocessingPipeline

    def test_import_filters(self):
        from src.preprocessing.filters import bandpass_filter, notch_filter, resample

    def test_import_artifacts(self):
        from src.preprocessing.artifacts import remove_artifacts_ica

    def test_import_reference(self):
        from src.preprocessing.reference import apply_car

    def test_import_normalization(self):
        from src.preprocessing.normalization import zscore_normalize, robust_normalize

    def test_import_epoching(self):
        from src.preprocessing.epoching import (
            create_event_epochs, create_sliding_window_epochs,
        )

    def test_import_quality(self):
        from src.preprocessing.quality import compute_quality_metrics


class TestImportsTraining:
    """Verify all training modules import."""

    def test_import_trainer(self):
        from src.training.trainer import Trainer

    def test_import_augmentation(self):
        from src.training.augmentation import (
            TimeShift, ChannelDropout, GaussianNoise,
            FrequencyMasking, TimeMasking, Compose,
        )

    def test_import_callbacks(self):
        from src.training.callbacks import EarlyStopping, ModelCheckpoint, GradientMonitor

    def test_import_schedulers(self):
        from src.training.schedulers import (
            CosineWarmupScheduler, LinearWarmupScheduler, build_scheduler,
        )

    def test_import_rl_trainer(self):
        from src.training.rl_trainer import RLTrainer, RewardComputer

    def test_import_hpo(self):
        from src.training.hpo import OptunaHPO, build_objective, OptunaPruningCallback

    def test_import_wandb_logger(self):
        from src.training.wandb_logger import WandbLogger, create_wandb_logger


class TestImportsEvaluation:
    """Verify all evaluation modules import."""

    def test_import_metrics(self):
        from src.evaluation.metrics import (
            compute_classification_metrics, compute_retrieval_metrics,
        )

    def test_import_statistical(self):
        from src.evaluation.statistical import (
            mcnemar_test, paired_permutation_test,
            corrected_paired_ttest, bonferroni_correction,
            compute_confidence_interval,
        )

    def test_import_visualization(self):
        from src.evaluation.visualization import (
            plot_confusion_matrix, plot_training_curves,
            plot_per_subject_performance, plot_ablation_comparison,
        )


class TestImportsDeployment:
    """Verify deployment modules import."""

    def test_import_exporter(self):
        from src.deployment.exporter import (
            export_onnx, export_torchscript, quantize_dynamic, profile_inference,
        )

    def test_import_streaming(self):
        from src.deployment.streaming import StreamingInference


class TestImportsUtils:
    """Verify utility modules import."""

    def test_import_config(self):
        from src.utils.config import load_config, merge_configs, save_config

    def test_import_seed(self):
        from src.utils.seed import seed_everything

    def test_import_device(self):
        from src.utils.device import DeviceManager

    def test_import_logging(self):
        from src.utils.logging import setup_logger

    def test_import_io(self):
        from src.utils.io import save_json, load_json, save_tsv


class TestImportsFeatures:
    """Verify feature modules import."""

    def test_import_region_pooling(self):
        from src.features.region_pooling import RegionPooling

    def test_import_datasets(self):
        from src.datasets.base import BaseEEGDataset
