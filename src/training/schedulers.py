"""
Learning rate schedulers — warmup + cosine, linear warmup, etc.
"""

from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineWarmupScheduler(_LRScheduler):
    """Cosine annealing with linear warmup.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer.
    warmup_epochs : int
        Number of warmup epochs.
    max_epochs : int
        Total number of epochs.
    min_lr : float
        Minimum learning rate at end of cosine.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.max_epochs - self.warmup_epochs
            )
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]


class LinearWarmupScheduler(_LRScheduler):
    """Linear warmup followed by constant learning rate.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    warmup_epochs : int
        Number of warmup epochs.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        return self.base_lrs


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    **kwargs,
) -> _LRScheduler | None:
    """Build scheduler from config.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    scheduler_name : str
        Scheduler type: 'cosine_warmup', 'linear_warmup', 'step', 'cosine', 'none'.
    **kwargs
        Additional scheduler arguments.

    Returns
    -------
    _LRScheduler or None
    """
    if scheduler_name == "cosine_warmup":
        return CosineWarmupScheduler(
            optimizer,
            warmup_epochs=kwargs.get("warmup_epochs", 10),
            max_epochs=kwargs.get("max_epochs", 200),
            min_lr=kwargs.get("min_lr", 1e-6),
        )
    elif scheduler_name == "linear_warmup":
        return LinearWarmupScheduler(
            optimizer,
            warmup_epochs=kwargs.get("warmup_epochs", 10),
        )
    elif scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 50),
            gamma=kwargs.get("gamma", 0.1),
        )
    elif scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get("max_epochs", 200),
            eta_min=kwargs.get("min_lr", 1e-6),
        )
    elif scheduler_name == "none" or scheduler_name is None:
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
