"""
Main Trainer — orchestrates training across all stages.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.utils.device import DeviceManager
from src.utils.seed import seed_everything

logger = logging.getLogger(__name__)


class Trainer:
    """Multi-stage trainer for NeuroGraph-Conformer.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    cfg : DictConfig
        Training configuration.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    device : str
        Device to train on.
    experiment_dir : str or Path
        Directory for logs and checkpoints.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: DictConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "auto",
        experiment_dir: str | Path = "experiments/default",
    ):
        self.cfg = cfg
        self.device_mgr = DeviceManager(device)
        self.device = self.device_mgr.device
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        self.optimizer = self._build_optimizer()

        # Scheduler
        self.scheduler = self._build_scheduler()

        # Loss functions
        self.criterion = self._build_criterion()

        # State
        self.current_epoch = 0
        self.best_metric = 0.0
        self.patience_counter = 0

        logger.info(f"Trainer initialized on {self.device_mgr.summary()}")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer from config."""
        cfg = self.cfg.training
        opt_name = cfg.get("optimizer", "adamw").lower()

        if hasattr(self.model, "get_param_groups"):
            params = self.model.get_param_groups(cfg.lr, cfg.weight_decay)
        else:
            params = self.model.parameters()

        if opt_name == "adamw":
            return torch.optim.AdamW(
                params, lr=cfg.lr, weight_decay=cfg.weight_decay,
                betas=(0.9, 0.999),
            )
        elif opt_name == "adam":
            return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        elif opt_name == "sgd":
            return torch.optim.SGD(
                params, lr=cfg.lr, weight_decay=cfg.weight_decay,
                momentum=0.9, nesterov=True,
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

    def _build_scheduler(self) -> Any:
        """Build learning rate scheduler."""
        cfg = self.cfg.training
        sched_name = cfg.get("scheduler", "cosine_warmup")

        if sched_name == "cosine_warmup":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.max_epochs - cfg.get("warmup_epochs", 10),
                eta_min=cfg.get("min_lr", 1e-6),
            )
        elif sched_name == "constant_with_warmup":
            return torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
        else:
            return None

    def _build_criterion(self) -> nn.Module:
        """Build loss function (basic CE; extended in subclasses)."""
        label_smoothing = self.cfg.training.get("label_smoothing", 0.1)
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def train_epoch(self) -> dict[str, float]:
        """Train for one epoch.

        Returns
        -------
        dict[str, float]
            Training metrics for this epoch.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(self.train_loader):
            eeg = batch["eeg"].to(self.device)
            labels = batch["label"].to(self.device)

            # Forward pass
            outputs = self.model(eeg, task="classification")
            logits = outputs["cls_logits"]

            loss = self.criterion(logits, labels)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            clip_val = self.cfg.training.get("gradient_clip_val", 1.0)
            if clip_val > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), clip_val)

            self.optimizer.step()

            # Metrics
            total_loss += loss.item() * eeg.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += eeg.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total

        return {"train_loss": avg_loss, "train_accuracy": accuracy}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Run validation.

        Returns
        -------
        dict[str, float]
            Validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in self.val_loader:
            eeg = batch["eeg"].to(self.device)
            labels = batch["label"].to(self.device)

            outputs = self.model(eeg, task="classification")
            logits = outputs["cls_logits"]

            loss = self.criterion(logits, labels)
            total_loss += loss.item() * eeg.size(0)

            all_preds.append(logits.argmax(dim=-1).cpu())
            all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        avg_loss = total_loss / len(all_labels)
        accuracy = (all_preds == all_labels).float().mean().item()

        # Balanced accuracy
        from sklearn.metrics import balanced_accuracy_score
        balanced_acc = balanced_accuracy_score(all_labels.numpy(), all_preds.numpy())

        return {
            "val_loss": avg_loss,
            "val_accuracy": accuracy,
            "val_balanced_accuracy": balanced_acc,
        }

    def fit(self) -> dict[str, list]:
        """Run full training loop.

        Returns
        -------
        dict[str, list]
            History of metrics per epoch.
        """
        cfg = self.cfg.training
        max_epochs = cfg.max_epochs
        patience = cfg.early_stopping.get("patience", 30)
        monitor = cfg.early_stopping.get("monitor", "val_balanced_accuracy")
        mode = cfg.early_stopping.get("mode", "max")

        history = {
            "train_loss": [], "train_accuracy": [],
            "val_loss": [], "val_accuracy": [], "val_balanced_accuracy": [],
        }

        logger.info(f"Starting training: {max_epochs} epochs, patience={patience}")

        for epoch in range(max_epochs):
            self.current_epoch = epoch
            t_start = time.time()

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Record history
            for k, v in {**train_metrics, **val_metrics}.items():
                if k in history:
                    history[k].append(v)

            elapsed = time.time() - t_start

            # Log
            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch+1}/{max_epochs} | "
                f"loss={train_metrics['train_loss']:.4f} | "
                f"val_bacc={val_metrics['val_balanced_accuracy']:.4f} | "
                f"lr={lr:.2e} | {elapsed:.1f}s"
            )

            # Early stopping
            current_metric = val_metrics[monitor]
            improved = (mode == "max" and current_metric > self.best_metric) or \
                       (mode == "min" and current_metric < self.best_metric)

            if improved:
                self.best_metric = current_metric
                self.patience_counter = 0
                self._save_checkpoint("best.pt", val_metrics)
                logger.info(f"  ✓ New best {monitor}: {self.best_metric:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1} (patience={patience})")
                    break

        # Save final checkpoint
        self._save_checkpoint("last.pt", val_metrics)

        return history

    def _save_checkpoint(self, filename: str, metrics: dict) -> None:
        """Save model checkpoint."""
        ckpt_dir = self.experiment_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "metrics": metrics,
            "config": dict(self.cfg),
        }

        torch.save(checkpoint, ckpt_dir / filename)
        logger.info(f"Saved checkpoint: {ckpt_dir / filename}")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_metric = checkpoint["best_metric"]
        logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")
