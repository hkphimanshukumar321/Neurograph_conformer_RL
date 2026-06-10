"""
Functional tests — Optuna HPO and W&B integration.

Verifies search space suggestion, objective construction,
config override application, and W&B logger API.
"""

import pytest
import numpy as np
import torch

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


@pytest.mark.skipif(not HAS_OPTUNA, reason="optuna not installed")
class TestOptunaSearchSpace:
    """Verify HPO search space definitions."""

    def test_model_hparams_keys(self):
        from src.training.hpo import suggest_model_hparams
        study = optuna.create_study()
        trial = study.ask()
        hparams = suggest_model_hparams(trial)
        assert "d_model" in hparams
        assert "n_layers" in hparams
        assert "n_heads" in hparams
        assert "frontend_type" in hparams

    def test_training_hparams_keys(self):
        from src.training.hpo import suggest_training_hparams
        study = optuna.create_study()
        trial = study.ask()
        hparams = suggest_training_hparams(trial)
        assert "lr" in hparams
        assert "batch_size" in hparams
        assert "dropout" in hparams
        assert "optimizer" in hparams

    def test_quick_hparams_keys(self):
        from src.training.hpo import suggest_quick_hparams
        study = optuna.create_study()
        trial = study.ask()
        hparams = suggest_quick_hparams(trial)
        assert "lr" in hparams
        assert "d_model" in hparams
        assert len(hparams) <= 10

    def test_lr_range(self):
        from src.training.hpo import suggest_training_hparams
        study = optuna.create_study()
        for _ in range(20):
            trial = study.ask()
            hparams = suggest_training_hparams(trial)
            assert 1e-5 <= hparams["lr"] <= 1e-2
            assert 0.05 <= hparams["dropout"] <= 0.5


@pytest.mark.skipif(not HAS_OPTUNA, reason="optuna not installed")
class TestOptunaObjective:
    """Verify objective function construction."""

    def test_build_objective_returns_callable(self):
        from src.training.hpo import build_objective
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"model": {"name": "test"}, "training": {"lr": 1e-3}})

        def dummy_train(trial_cfg, trial):
            return np.random.random()

        objective = build_objective(cfg, dummy_train, search_space="quick")
        assert callable(objective)

    def test_optuna_study_runs(self):
        """Verify a minimal HPO study can execute."""
        from src.training.hpo import build_objective
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"model": {"name": "test"}, "training": {"lr": 1e-3}})

        def dummy_train(trial_cfg, trial):
            return np.random.random()

        objective = build_objective(cfg, dummy_train, search_space="quick")

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=3, show_progress_bar=False)

        assert len(study.trials) == 3
        assert study.best_trial is not None

    def test_pruning_callback(self):
        from src.training.hpo import OptunaPruningCallback

        study = optuna.create_study()
        trial = study.ask()

        cb = OptunaPruningCallback(trial, monitor="val_acc")
        # Should not prune early
        cb(epoch=0, metrics={"val_acc": 0.5})
        cb(epoch=1, metrics={"val_acc": 0.6})


class TestWandbLogger:
    """Verify W&B logger API (without actually logging to W&B)."""

    def test_disabled_mode(self):
        from src.training.wandb_logger import WandbLogger
        logger = WandbLogger(mode="disabled", experiment_name="test")
        # Should not raise even when disabled
        logger.log_metrics({"loss": 0.5}, step=0)
        logger.log_epoch(0, {"loss": 0.5}, {"loss": 0.4})
        logger.finish()

    def test_create_wandb_logger(self):
        from src.training.wandb_logger import create_wandb_logger
        # Should not raise even if wandb is not configured
        logger = create_wandb_logger(
            cfg={"model": "test"},
            experiment_name="unit_test",
            tags=["test"],
        )
        assert logger is not None
