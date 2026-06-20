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


def plot_multitask_losses(
    history: dict[str, list[float]],
    title: str = "Multi-Task Loss Breakdown",
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (16, 10),
) -> Any:
    """Plot per-component loss curves and embedding quality over training.

    Shows how each loss component (classification, contrastive, total)
    evolves, plus silhouette score as a measure of the contrastive
    reward's effect on embedding space organization.

    Parameters
    ----------
    history : dict[str, list[float]]
        Training history with keys like 'train_cls_loss', etc.
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

    has_silhouette = "val_silhouette_score" in history and any(
        v != 0 for v in history.get("val_silhouette_score", [])
    )
    n_panels = 4 if has_silhouette else 3

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)

    # Panel 1: Training losses breakdown
    ax = axes[0]
    if "train_loss" in history and history["train_loss"]:
        epochs = range(1, len(history["train_loss"]) + 1)
        ax.plot(epochs, history["train_loss"], label="Total Loss",
                linewidth=2, color="#2196F3")
    if "train_cls_loss" in history and history["train_cls_loss"]:
        ax.plot(epochs, history["train_cls_loss"], label="Classification",
                linewidth=1.5, linestyle="--", color="#4CAF50")
    if "train_contrast_loss" in history and history["train_contrast_loss"]:
        ax.plot(epochs, history["train_contrast_loss"],
                label="Contrastive (SupCon)",
                linewidth=1.5, linestyle="--", color="#FF9800")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Losses")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Validation losses breakdown
    ax = axes[1]
    if "val_loss" in history and history["val_loss"]:
        epochs = range(1, len(history["val_loss"]) + 1)
        ax.plot(epochs, history["val_loss"], label="Total Loss",
                linewidth=2, color="#2196F3")
    if "val_cls_loss" in history and history["val_cls_loss"]:
        ax.plot(epochs, history["val_cls_loss"], label="Classification",
                linewidth=1.5, linestyle="--", color="#4CAF50")
    if "val_contrast_loss" in history and history["val_contrast_loss"]:
        ax.plot(epochs, history["val_contrast_loss"],
                label="Contrastive (SupCon)",
                linewidth=1.5, linestyle="--", color="#FF9800")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Validation Losses")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Loss weight contribution (stacked area)
    ax = axes[2]
    if "train_cls_loss" in history and history["train_cls_loss"]:
        epochs = range(1, len(history["train_cls_loss"]) + 1)
        cls_vals = np.array(history["train_cls_loss"])
        contrast_vals = np.array(
            history.get("train_contrast_loss", [0] * len(cls_vals))
        )
        total = cls_vals + contrast_vals
        cls_pct = np.where(total > 0, cls_vals / total * 100, 50)
        con_pct = np.where(total > 0, contrast_vals / total * 100, 50)
        ax.stackplot(
            epochs, cls_pct, con_pct,
            labels=["Classification %", "Contrastive %"],
            colors=["#4CAF50", "#FF9800"], alpha=0.7,
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss Contribution (%)")
        ax.set_title("Loss Balance")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

    # Panel 4: Silhouette score (embedding quality)
    if has_silhouette:
        ax = axes[3]
        epochs = range(1, len(history["val_silhouette_score"]) + 1)
        ax.plot(
            epochs, history["val_silhouette_score"],
            linewidth=2, color="#9C27B0", marker="o", markersize=3,
        )
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Silhouette Score")
        ax.set_title("Embedding Cluster Quality")
        ax.grid(True, alpha=0.3)
        ax.annotate(
            "Good separation →", xy=(0.05, 0.95),
            xycoords="axes fraction", fontsize=8, color="green", va="top",
        )
        ax.annotate(
            "← Overlapping", xy=(0.05, 0.05),
            xycoords="axes fraction", fontsize=8, color="red", va="bottom",
        )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Multi-task loss plot saved to {save_path}")

    return fig


def plot_reward_curves(
    history: dict[str, list[float]],
    title: str = "RL Reward Progress",
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (16, 5),
) -> Any:
    """Plot RL reward metrics over training epochs.

    Shows how the SCST reward signal improves during RL fine-tuning:

    - **Greedy reward** (baseline): model's best deterministic output quality
    - **Sample reward**: quality of exploratory outputs
    - **Advantage**: whether exploration finds better outputs than greedy
    - **RL loss**: the policy gradient loss magnitude

    Parameters
    ----------
    history : dict[str, list[float]]
        Training history with RL-specific keys.
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

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel 1: Rewards over time
    ax = axes[0]
    has_data = False
    if "rl_greedy_reward" in history and history["rl_greedy_reward"]:
        epochs = range(1, len(history["rl_greedy_reward"]) + 1)
        ax.plot(
            epochs, history["rl_greedy_reward"],
            label="Greedy Reward (baseline)",
            linewidth=2, color="#2196F3",
        )
        has_data = True
    if "rl_sample_reward" in history and history["rl_sample_reward"]:
        ax.plot(
            epochs, history["rl_sample_reward"],
            label="Sample Reward (exploration)",
            linewidth=2, color="#FF9800", alpha=0.7,
        )
        has_data = True
    if has_data:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Reward")
        ax.set_title("Reward Over Training")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Panel 2: Advantage (sample - greedy)
    ax = axes[1]
    if "rl_advantage" in history and history["rl_advantage"]:
        epochs = range(1, len(history["rl_advantage"]) + 1)
        advantages = history["rl_advantage"]
        colors = ["#4CAF50" if a >= 0 else "#F44336" for a in advantages]
        ax.bar(epochs, advantages, color=colors, alpha=0.7, width=0.8)
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Advantage")
        ax.set_title(
            "Exploration Advantage\n"
            "(positive = exploration found better outputs)"
        )
        ax.grid(True, axis="y", alpha=0.3)

    # Panel 3: RL loss magnitude
    ax = axes[2]
    if "rl_loss" in history and history["rl_loss"]:
        epochs = range(1, len(history["rl_loss"]) + 1)
        ax.plot(epochs, history["rl_loss"], linewidth=2, color="#9C27B0")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("RL Loss")
        ax.set_title("SCST Policy Gradient Loss")
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"RL reward curves saved to {save_path}")

    return fig
