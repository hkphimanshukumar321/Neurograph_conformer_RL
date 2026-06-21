"""
Weights & Biases (W&B) integration for experiment tracking.

Provides a unified WandbLogger that handles:
  - Run initialization with project/entity/config
  - Metric logging (train/val per epoch)
  - Hyperparameter tracking
  - Model artifact logging (checkpoints)
  - System metrics (GPU utilization, memory)
  - Table logging for per-subject results
  - Alert notifications for best metrics
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ──── W&B API key setup ────
# Set the API key as environment variable for seamless authentication
os.environ["WANDB_API_KEY"] = "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"


class WandbLogger:
    """Weights & Biases logger for NeuroGraph-Conformer experiments.

    Parameters
    ----------
    project : str
        W&B project name.
    experiment_name : str
        Run name / experiment identifier.
    config : dict, optional
        Hyperparameters and config to log.
    entity : str, optional
        W&B team/entity name.
    tags : list[str], optional
        Tags for the run.
    group : str, optional
        Group name (e.g., ablation study ID).
    job_type : str, optional
        Job type (e.g., 'train', 'eval', 'hpo').
    resume : bool
        Whether to resume a previous run.
    mode : str
        'online', 'offline', or 'disabled'.
    save_code : bool
        Whether to save the code with the run.
    """

    def __init__(
        self,
        project: str = "NeuroGraph-Conformer-RL",
        experiment_name: str = "default",
        config: dict | None = None,
        entity: str | None = None,
        tags: list[str] | None = None,
        group: str | None = None,
        job_type: str = "train",
        resume: bool = False,
        mode: str = "online",
        save_code: bool = True,
    ):
        self.project = project
        self.experiment_name = experiment_name
        self._run = None
        self._enabled = True

        try:
            import wandb

            self._run = wandb.init(
                project=project,
                name=experiment_name,
                config=config or {},
                entity=entity,
                tags=tags or [],
                group=group,
                job_type=job_type,
                resume="allow" if resume else None,
                mode=mode,
                save_code=save_code,
                reinit=True,
            )

            logger.info(
                f"W&B initialized: project={project}, run={experiment_name}, "
                f"url={self._run.url}"
            )

        except ImportError:
            logger.warning("wandb not installed. Install: pip install wandb")
            self._enabled = False
        except Exception as e:
            logger.warning(f"W&B init failed: {e}. Continuing without W&B.")
            self._enabled = False

    @property
    def run(self):
        """Get the active W&B run."""
        return self._run

    @property
    def enabled(self) -> bool:
        return self._enabled and self._run is not None

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
        commit: bool = True,
    ) -> None:
        """Log metrics to W&B.

        Parameters
        ----------
        metrics : dict
            Metric name → value pairs.
        step : int, optional
            Global step (epoch number).
        commit : bool
            Whether to commit the log immediately.
        """
        if not self.enabled:
            return

        import wandb

        if step is not None:
            metrics["epoch"] = step

        wandb.log(metrics, step=step, commit=commit)

    def log_epoch(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        lr: float | None = None,
    ) -> None:
        """Log a complete epoch's metrics.

        Parameters
        ----------
        epoch : int
            Epoch number.
        train_metrics : dict
            Training metrics.
        val_metrics : dict
            Validation metrics.
        lr : float, optional
            Current learning rate.
        """
        if not self.enabled:
            return

        combined = {}
        for k, v in train_metrics.items():
            combined[f"train/{k}"] = v
        for k, v in val_metrics.items():
            combined[f"val/{k}"] = v
        if lr is not None:
            combined["lr"] = lr

        self.log_metrics(combined, step=epoch)

    def log_config(self, config: dict) -> None:
        """Update run config.

        Parameters
        ----------
        config : dict
            Configuration to merge.
        """
        if not self.enabled:
            return

        import wandb
        wandb.config.update(config, allow_val_change=True)

    def log_model(
        self,
        model_path: str | Path,
        name: str = "model",
        metadata: dict | None = None,
    ) -> None:
        """Log a model checkpoint as a W&B artifact.

        Parameters
        ----------
        model_path : str or Path
            Path to the model checkpoint file.
        name : str
            Artifact name.
        metadata : dict, optional
            Metadata to attach.
        """
        if not self.enabled:
            return

        import wandb

        artifact = wandb.Artifact(
            name=name,
            type="model",
            metadata=metadata or {},
        )
        artifact.add_file(str(model_path))
        self._run.log_artifact(artifact)
        logger.info(f"Model artifact logged: {name}")

    def log_table(
        self,
        table_name: str,
        columns: list[str],
        data: list[list[Any]],
    ) -> None:
        """Log a table (e.g., per-subject results).

        Parameters
        ----------
        table_name : str
            Table name.
        columns : list[str]
            Column names.
        data : list[list]
            Table data rows.
        """
        if not self.enabled:
            return

        import wandb

        table = wandb.Table(columns=columns, data=data)
        self._run.log({table_name: table})

    def log_confusion_matrix(
        self,
        y_true: list[int],
        y_pred: list[int],
        class_names: list[str] | None = None,
    ) -> None:
        """Log a confusion matrix.

        Parameters
        ----------
        y_true : list[int]
            Ground truth labels.
        y_pred : list[int]
            Predicted labels.
        class_names : list[str], optional
            Class names.
        """
        if not self.enabled:
            return

        import wandb

        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                y_true=y_true,
                preds=y_pred,
                class_names=class_names,
            )
        })

    def log_histogram(
        self,
        key: str,
        values: list[float] | Any,
        step: int | None = None,
    ) -> None:
        """Log a histogram."""
        if not self.enabled:
            return

        import wandb
        wandb.log({key: wandb.Histogram(values)}, step=step)

    def log_image(
        self,
        key: str,
        image_path: str | Path,
        caption: str = "",
    ) -> None:
        """Log an image (e.g., confusion matrix plot)."""
        if not self.enabled:
            return

        import wandb
        wandb.log({key: wandb.Image(str(image_path), caption=caption)})

    def alert(
        self,
        title: str,
        text: str,
        level: str = "INFO",
    ) -> None:
        """Send a W&B alert.

        Parameters
        ----------
        title : str
            Alert title.
        text : str
            Alert message.
        level : str
            Alert level: 'INFO', 'WARN', 'ERROR'.
        """
        if not self.enabled:
            return

        import wandb

        level_map = {
            "INFO": wandb.AlertLevel.INFO,
            "WARN": wandb.AlertLevel.WARN,
            "ERROR": wandb.AlertLevel.ERROR,
        }
        wandb.alert(title=title, text=text, level=level_map.get(level, wandb.AlertLevel.INFO))

    def watch_model(
        self,
        model: Any,
        log: str = "gradients",
        log_freq: int = 100,
    ) -> None:
        """Watch model gradients/parameters.

        Parameters
        ----------
        model : nn.Module
            PyTorch model.
        log : str
            What to log: 'gradients', 'parameters', 'all'.
        log_freq : int
            Logging frequency in steps.
        """
        if not self.enabled:
            return

        import wandb
        wandb.watch(model, log=log, log_freq=log_freq)

    def finish(self) -> None:
        """Finish the W&B run."""
        if self.enabled:
            import wandb
            wandb.finish()
            logger.info("W&B run finished")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
        return False


# ──── Convenience Functions ────

def create_wandb_logger(
    cfg: dict,
    experiment_name: str,
    tags: list[str] | None = None,
) -> WandbLogger:
    """Create a WandbLogger from config.

    Parameters
    ----------
    cfg : dict
        Full experiment config (will be logged).
    experiment_name : str
        Experiment name.
    tags : list[str], optional
        Run tags.

    Returns
    -------
    WandbLogger
    """
    return WandbLogger(
        project="NeuroGraph-Conformer-RL",
        experiment_name=experiment_name,
        config=cfg,
        tags=tags or [],
        job_type="train",
    )
