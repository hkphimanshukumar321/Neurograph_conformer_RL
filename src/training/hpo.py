"""
Optuna-based hyperparameter optimization for NeuroGraph-Conformer.

Supports:
  - Model architecture search (d_model, n_layers, n_heads, etc.)
  - Training hyperparameters (lr, weight_decay, dropout, etc.)
  - Front-end selection (CWT vs FilterBank)
  - Pruning of unpromising trials via MedianPruner
  - W&B integration for tracking
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import optuna
from optuna.integration import WeightsAndBiasesCallback
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.samplers import TPESampler
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


# ──── Search Space Definitions ────

def suggest_model_hparams(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest model architecture hyperparameters.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object.

    Returns
    -------
    dict
        Suggested hyperparameters.
    """
    hparams = {}

    # Frontend
    hparams["frontend_type"] = trial.suggest_categorical(
        "frontend_type", ["cwt", "filterbank"]
    )

    if hparams["frontend_type"] == "cwt":
        hparams["cwt_time_downsample"] = trial.suggest_categorical(
            "cwt_time_downsample", [2, 4, 8]
        )
    else:
        hparams["filterbank_n_bands"] = trial.suggest_int(
            "filterbank_n_bands", 4, 9
        )

    # Spatial encoder
    hparams["spatial_type"] = trial.suggest_categorical(
        "spatial_type", ["graph", "region_pooling", "none"]
    )

    if hparams["spatial_type"] == "graph":
        hparams["graph_n_layers"] = trial.suggest_int("graph_n_layers", 1, 3)
        hparams["graph_n_heads"] = trial.suggest_categorical("graph_n_heads", [2, 4, 8])
        hparams["adjacency_type"] = trial.suggest_categorical(
            "adjacency_type", ["distance", "correlation", "hybrid"]
        )

    # Conformer encoder
    hparams["d_model"] = trial.suggest_categorical("d_model", [64, 128, 256])
    hparams["n_layers"] = trial.suggest_int("n_layers", 2, 8)
    hparams["n_heads"] = trial.suggest_categorical("n_heads", [4, 8])
    hparams["d_ff"] = trial.suggest_categorical("d_ff", [256, 512, 1024])
    hparams["conv_kernel_size"] = trial.suggest_categorical(
        "conv_kernel_size", [15, 31, 63]
    )
    hparams["macaron"] = trial.suggest_categorical("macaron", [True, False])

    # Mamba
    hparams["use_mamba"] = trial.suggest_categorical("use_mamba", [True, False])
    if hparams["use_mamba"]:
        hparams["mamba_n_layers"] = trial.suggest_int("mamba_n_layers", 1, 3)
        hparams["mamba_d_state"] = trial.suggest_categorical("mamba_d_state", [8, 16, 32])

    return hparams


def suggest_training_hparams(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest training hyperparameters.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object.

    Returns
    -------
    dict
        Suggested hyperparameters.
    """
    hparams = {}

    hparams["lr"] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    hparams["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    hparams["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64])
    hparams["dropout"] = trial.suggest_float("dropout", 0.05, 0.5)
    hparams["attention_dropout"] = trial.suggest_float("attention_dropout", 0.0, 0.3)
    hparams["label_smoothing"] = trial.suggest_float("label_smoothing", 0.0, 0.2)
    hparams["gradient_clip_val"] = trial.suggest_categorical(
        "gradient_clip_val", [0.5, 1.0, 5.0]
    )
    hparams["warmup_epochs"] = trial.suggest_int("warmup_epochs", 3, 15)
    hparams["optimizer"] = trial.suggest_categorical("optimizer", ["adamw", "adam"])

    # Augmentation
    hparams["mixup_alpha"] = trial.suggest_float("mixup_alpha", 0.0, 0.4)
    hparams["channel_dropout_p"] = trial.suggest_float("channel_dropout_p", 0.0, 0.2)
    hparams["noise_std"] = trial.suggest_float("noise_std", 0.0, 0.2)

    return hparams


def suggest_quick_hparams(trial: optuna.Trial) -> dict[str, Any]:
    """Reduced search space for quick experiments.

    Parameters
    ----------
    trial : optuna.Trial

    Returns
    -------
    dict
    """
    return {
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.4),
        "d_model": trial.suggest_categorical("d_model", [64, 128]),
        "n_layers": trial.suggest_int("n_layers", 2, 6),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
    }


# ──── Objective Builder ────

def build_objective(
    cfg: DictConfig,
    train_fn: Callable,
    search_space: str = "full",
) -> Callable[[optuna.Trial], float]:
    """Build an Optuna objective function.

    Parameters
    ----------
    cfg : DictConfig
        Base configuration.
    train_fn : Callable
        Training function: (cfg, trial) → metric_value.
    search_space : str
        Search space type: 'full', 'training_only', 'quick'.

    Returns
    -------
    Callable
        Optuna objective function.
    """

    def objective(trial: optuna.Trial) -> float:
        # Suggest hyperparameters
        if search_space == "full":
            model_hparams = suggest_model_hparams(trial)
            training_hparams = suggest_training_hparams(trial)
            hparams = {**model_hparams, **training_hparams}
        elif search_space == "training_only":
            hparams = suggest_training_hparams(trial)
        elif search_space == "quick":
            hparams = suggest_quick_hparams(trial)
        else:
            raise ValueError(f"Unknown search space: {search_space}")

        # Override config with suggested hparams
        trial_cfg = OmegaConf.create(dict(cfg))
        _apply_hparams(trial_cfg, hparams)

        # Train and return validation metric
        try:
            metric = train_fn(trial_cfg, trial)
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()

        return metric

    return objective


def _apply_hparams(cfg: DictConfig, hparams: dict) -> None:
    """Apply suggested hyperparameters to config."""
    mapping = {
        # Model
        "d_model": "model.arch.encoder.d_model",
        "n_layers": "model.arch.encoder.n_layers",
        "n_heads": "model.arch.encoder.n_heads",
        "d_ff": "model.arch.encoder.d_ff",
        "conv_kernel_size": "model.arch.encoder.conv_kernel_size",
        "macaron": "model.arch.encoder.macaron",
        "frontend_type": "model.arch.frontend.type",
        "spatial_type": "model.arch.spatial.type",
        "use_mamba": "model.arch.mamba.enabled",
        # Training
        "lr": "training.lr",
        "weight_decay": "training.weight_decay",
        "batch_size": "training.batch_size",
        "dropout": "model.arch.encoder.dropout",
        "attention_dropout": "model.arch.encoder.attention_dropout",
        "label_smoothing": "training.losses.cross_entropy.label_smoothing",
        "gradient_clip_val": "training.gradient_clip_val",
        "warmup_epochs": "training.warmup_epochs",
        "optimizer": "training.optimizer",
    }

    for key, path in mapping.items():
        if key in hparams:
            OmegaConf.update(cfg, path, hparams[key], force_add=True)


# ──── Study Runner ────

class OptunaHPO:
    """Optuna hyperparameter optimization manager.

    Parameters
    ----------
    study_name : str
        Name of the Optuna study.
    direction : str
        Optimization direction: 'maximize' or 'minimize'.
    n_trials : int
        Number of trials to run.
    storage : str, optional
        Optuna storage URL (e.g., 'sqlite:///optuna.db').
    pruner : str
        Pruner type: 'median', 'hyperband', 'none'.
    wandb_project : str, optional
        W&B project name for logging.
    """

    def __init__(
        self,
        study_name: str = "neurograph_hpo",
        direction: str = "maximize",
        n_trials: int = 100,
        storage: str | None = None,
        pruner: str = "median",
        wandb_project: str | None = None,
    ):
        self.study_name = study_name
        self.direction = direction
        self.n_trials = n_trials
        self.wandb_project = wandb_project

        # Build pruner
        if pruner == "median":
            optuna_pruner = MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1,
            )
        elif pruner == "hyperband":
            optuna_pruner = HyperbandPruner(
                min_resource=5,
                max_resource=200,
                reduction_factor=3,
            )
        else:
            optuna_pruner = optuna.pruners.NopPruner()

        # Build sampler
        sampler = TPESampler(
            seed=42,
            n_startup_trials=10,
            multivariate=True,
        )

        # Create or load study
        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            pruner=optuna_pruner,
            storage=storage,
            load_if_exists=True,
        )

        logger.info(
            f"Optuna study '{study_name}' created/loaded "
            f"(direction={direction}, n_trials={n_trials})"
        )

    def run(
        self,
        objective: Callable[[optuna.Trial], float],
        n_jobs: int = 1,
        timeout: int | None = None,
    ) -> optuna.Study:
        """Run the optimization.

        Parameters
        ----------
        objective : Callable
            Objective function.
        n_jobs : int
            Number of parallel workers.
        timeout : int, optional
            Time limit in seconds.

        Returns
        -------
        optuna.Study
            Completed study.
        """
        callbacks = []

        # W&B callback
        if self.wandb_project:
            try:
                wandb_callback = WeightsAndBiasesCallback(
                    metric_name="val_balanced_accuracy",
                    wandb_kwargs={
                        "project": self.wandb_project,
                        "group": self.study_name,
                    },
                )
                callbacks.append(wandb_callback)
                logger.info(f"W&B callback enabled (project={self.wandb_project})")
            except Exception as e:
                logger.warning(f"W&B callback failed: {e}")

        # Run optimization
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=n_jobs,
            timeout=timeout,
            callbacks=callbacks,
            show_progress_bar=True,
        )

        # Report results
        self._report_results()

        return self.study

    def _report_results(self) -> None:
        """Log study results."""
        study = self.study

        logger.info(f"\n{'='*60}")
        logger.info(f"HPO Study: {self.study_name}")
        logger.info(f"{'='*60}")
        logger.info(f"Total trials: {len(study.trials)}")
        logger.info(f"Completed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        logger.info(f"Pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        logger.info(f"Failed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")

        if study.best_trial:
            logger.info(f"\nBest trial: #{study.best_trial.number}")
            logger.info(f"Best value: {study.best_trial.value:.4f}")
            logger.info("Best params:")
            for key, value in study.best_trial.params.items():
                logger.info(f"  {key}: {value}")

    def get_best_config(self, base_cfg: DictConfig) -> DictConfig:
        """Get the best configuration from the study.

        Parameters
        ----------
        base_cfg : DictConfig
            Base configuration to override.

        Returns
        -------
        DictConfig
            Configuration with best hyperparameters.
        """
        best_cfg = OmegaConf.create(dict(base_cfg))
        if self.study.best_trial:
            _apply_hparams(best_cfg, self.study.best_trial.params)
        return best_cfg

    def save_results(self, output_dir: str | Path) -> None:
        """Save study results and visualization.

        Parameters
        ----------
        output_dir : str or Path
            Output directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save best params as YAML
        if self.study.best_trial:
            best_params = OmegaConf.create(self.study.best_trial.params)
            OmegaConf.save(best_params, output_dir / "best_params.yaml")

        # Save all trials as CSV
        try:
            df = self.study.trials_dataframe()
            df.to_csv(output_dir / "all_trials.csv", index=False)
        except Exception:
            pass

        # Save visualizations
        try:
            from optuna.visualization import (
                plot_optimization_history,
                plot_param_importances,
                plot_parallel_coordinate,
                plot_contour,
            )

            fig = plot_optimization_history(self.study)
            fig.write_html(str(output_dir / "optimization_history.html"))

            fig = plot_param_importances(self.study)
            fig.write_html(str(output_dir / "param_importances.html"))

            fig = plot_parallel_coordinate(self.study)
            fig.write_html(str(output_dir / "parallel_coordinate.html"))

            logger.info(f"Visualizations saved to {output_dir}")
        except Exception as e:
            logger.warning(f"Failed to save visualizations: {e}")


# ──── Pruning Callback for Trainer ────

class OptunaPruningCallback:
    """Report intermediate values to Optuna for trial pruning.

    Use this in the training loop to enable early stopping of
    unpromising trials.

    Parameters
    ----------
    trial : optuna.Trial
        The Optuna trial.
    monitor : str
        Metric name to report.
    """

    def __init__(self, trial: optuna.Trial, monitor: str = "val_balanced_accuracy"):
        self.trial = trial
        self.monitor = monitor

    def __call__(self, epoch: int, metrics: dict[str, float]) -> None:
        """Report metric and check for pruning.

        Parameters
        ----------
        epoch : int
            Current epoch.
        metrics : dict
            Validation metrics.

        Raises
        ------
        optuna.TrialPruned
            If the trial should be pruned.
        """
        value = metrics.get(self.monitor)
        if value is not None:
            self.trial.report(value, epoch)
            if self.trial.should_prune():
                raise optuna.TrialPruned()
