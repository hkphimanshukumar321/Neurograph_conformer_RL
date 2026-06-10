"""
Training callbacks — early stopping, model checkpointing, learning rate logging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping callback.

    Parameters
    ----------
    patience : int
        Number of epochs without improvement before stopping.
    monitor : str
        Metric name to monitor.
    mode : str
        'min' or 'max'.
    min_delta : float
        Minimum change to qualify as improvement.
    """

    def __init__(
        self,
        patience: int = 30,
        monitor: str = "val_balanced_accuracy",
        mode: str = "max",
        min_delta: float = 1e-4,
    ):
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        """Check if training should stop.

        Returns True if training should stop.
        """
        if self.mode == "min":
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping triggered after {self.counter} epochs "
                    f"without improvement on {self.monitor}"
                )

        return self.should_stop


class ModelCheckpoint:
    """Save model checkpoints.

    Parameters
    ----------
    save_dir : str or Path
        Directory to save checkpoints.
    monitor : str
        Metric to monitor for best model.
    mode : str
        'min' or 'max'.
    save_last : bool
        Whether to save last epoch checkpoint.
    save_top_k : int
        Number of best checkpoints to keep.
    """

    def __init__(
        self,
        save_dir: str | Path,
        monitor: str = "val_balanced_accuracy",
        mode: str = "max",
        save_last: bool = True,
        save_top_k: int = 3,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_last = save_last
        self.save_top_k = save_top_k
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.top_checkpoints: list[tuple[float, Path]] = []

    def __call__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Save checkpoint if metrics improved."""
        current = metrics.get(self.monitor, 0)

        if self.mode == "min":
            improved = current < self.best_value
        else:
            improved = current > self.best_value

        if improved:
            self.best_value = current
            path = self.save_dir / "best.pt"
            self._save(model, optimizer, epoch, metrics, path)
            logger.info(f"New best {self.monitor}: {current:.4f} → saved to {path}")

        # Save top-k
        path = self.save_dir / f"epoch_{epoch:04d}.pt"
        self._save(model, optimizer, epoch, metrics, path)
        self.top_checkpoints.append((current, path))
        self.top_checkpoints.sort(
            key=lambda x: x[0],
            reverse=(self.mode == "max"),
        )
        # Remove excess checkpoints
        while len(self.top_checkpoints) > self.save_top_k:
            _, old_path = self.top_checkpoints.pop()
            if old_path.exists() and "best" not in old_path.name:
                old_path.unlink()

        if self.save_last:
            self._save(model, optimizer, epoch, metrics, self.save_dir / "last.pt")

    @staticmethod
    def _save(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict,
        path: Path,
    ) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
            },
            path,
        )


class GradientMonitor:
    """Monitor gradient norms during training for debugging.

    Parameters
    ----------
    log_interval : int
        Log gradient stats every N steps.
    """

    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.step = 0

    def __call__(self, model: nn.Module) -> dict[str, float]:
        """Log gradient statistics."""
        self.step += 1
        if self.step % self.log_interval != 0:
            return {}

        total_norm = 0.0
        max_norm = 0.0
        param_count = 0

        for name, p in model.named_parameters():
            if p.grad is not None:
                norm = p.grad.data.norm(2).item()
                total_norm += norm ** 2
                max_norm = max(max_norm, norm)
                param_count += 1

        total_norm = total_norm ** 0.5
        stats = {
            "grad/total_norm": total_norm,
            "grad/max_norm": max_norm,
            "grad/n_params_with_grad": param_count,
        }

        if total_norm > 100:
            logger.warning(f"Large gradient norm: {total_norm:.2f}")

        return stats
