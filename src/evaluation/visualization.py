"""
Visualization utilities for evaluation and paper figures.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
    normalize: bool = True,
    cmap: str = "Blues",
    figsize: tuple[int, int] = (10, 8),
) -> Any:
    """Plot confusion matrix with annotations.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix, shape (n_classes, n_classes).
    class_names : list[str]
        Class labels.
    title : str
        Plot title.
    save_path : str or Path, optional
        Path to save figure.
    normalize : bool
        Whether to normalize rows to percentages.
    cmap : str
        Colormap name.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not installed — skipping plot")
        return None

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = cm.astype(float) / np.maximum(row_sums, 1) * 100
        fmt = ".1f"
    else:
        cm_plot = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        square=True,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {save_path}")

    return fig


def plot_training_curves(
    history: dict[str, list[float]],
    title: str = "Training History",
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (14, 5),
) -> Any:
    """Plot training and validation curves.

    Parameters
    ----------
    history : dict[str, list[float]]
        Training history with keys like 'train_loss', 'val_accuracy', etc.
    title : str
        Plot title.
    save_path : str or Path, optional
        Path to save figure.
    figsize : tuple
        Figure size.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot")
        return None

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Loss
    if "train_loss" in history:
        axes[0].plot(history["train_loss"], label="Train Loss", alpha=0.8)
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="Val Loss", alpha=0.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    if "train_accuracy" in history:
        axes[1].plot(history["train_accuracy"], label="Train Acc", alpha=0.8)
    if "val_accuracy" in history:
        axes[1].plot(history["val_accuracy"], label="Val Acc", alpha=0.8)
    if "val_balanced_accuracy" in history:
        axes[1].plot(
            history["val_balanced_accuracy"], label="Val Balanced Acc",
            alpha=0.8, linestyle="--",
        )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Training curves saved to {save_path}")

    return fig


def plot_per_subject_performance(
    subject_ids: list[str],
    accuracies: list[float],
    title: str = "Per-Subject Performance",
    save_path: str | Path | None = None,
) -> Any:
    """Plot per-subject accuracy bar chart for LOSO analysis."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(subject_ids))
    bars = ax.bar(x, accuracies, color="steelblue", alpha=0.8)

    # Color bars below chance differently
    mean_acc = np.mean(accuracies)
    for bar, acc in zip(bars, accuracies):
        if acc < mean_acc * 0.5:
            bar.set_color("salmon")

    ax.axhline(y=mean_acc, color="red", linestyle="--", label=f"Mean: {mean_acc:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(subject_ids, rotation=45, ha="right")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_ablation_comparison(
    ablation_names: list[str],
    scores: dict[str, list[float]],
    title: str = "Ablation Study",
    save_path: str | Path | None = None,
) -> Any:
    """Plot ablation results with error bars."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(ablation_names))
    width = 0.35

    for i, (metric_name, values) in enumerate(scores.items()):
        means = [np.mean(v) if isinstance(v, list) else v for v in values]
        stds = [np.std(v) if isinstance(v, list) else 0 for v in values]
        ax.bar(
            x + i * width, means, width,
            yerr=stds, label=metric_name,
            alpha=0.8, capsize=3,
        )

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(ablation_names, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
