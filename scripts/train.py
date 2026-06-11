"""
train.py — Main training entry point.

Usage:
    python scripts/train.py --config configs/models/conformer_small.yaml \
        --dataset chisco --protocol within_subject --experiment E1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config, merge_configs
from src.utils.logging import setup_logger
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NeuroGraph-Conformer Training")
    parser.add_argument("--config", type=str, required=True, help="Model config YAML")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["thinking_out_loud", "chisco", "zuco"],
                        help="Dataset name")
    parser.add_argument("--protocol", type=str, default="within_subject",
                        choices=["within_subject", "loso", "cross_dataset"],
                        help="Evaluation protocol")
    parser.add_argument("--experiment", type=str, default="default",
                        help="Experiment name for logging")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained checkpoint")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overrides", nargs="*", default=[],
                        help="Config overrides in dot notation")
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup
    logger = setup_logger("neurograph", level="INFO")
    seed_everything(args.seed)

    # Load configs
    base_cfg = load_config("configs/base.yaml")
    dataset_cfg = load_config(f"configs/datasets/{args.dataset}.yaml")
    model_cfg = load_config(args.config)
    cfg = merge_configs(base_cfg, dataset_cfg, model_cfg)

    # Apply CLI overrides
    if args.overrides:
        from omegaconf import OmegaConf
        cli_cfg = OmegaConf.from_dotlist(args.overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    logger.info(f"Experiment: {args.experiment}")
    logger.info(f"Dataset: {args.dataset}, Protocol: {args.protocol}")
    logger.info(f"Model: {cfg.model.name}")

    # Experiment directory
    exp_dir = Path("experiments") / args.experiment
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    from src.utils.config import save_config
    save_config(cfg, exp_dir / "config.yaml")

    # ── Build dataset ──
    from src.datasets.factory import get_dataset

    logger.info(f"Loading {args.dataset} dataset...")
    manifest_path = Path("data/processed/manifests/trials.csv")

    if not manifest_path.exists():
        logger.error(f"Manifest not found at {manifest_path}. Run preprocessing first.")
        return

    full_dataset = get_dataset(args.dataset, manifest_path=manifest_path, split="train")

    if len(full_dataset) == 0:
        logger.error(f"No samples found for {args.dataset} in {manifest_path}")
        return

    # Split 80/20
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    logger.info(f"Loaded {train_size} training samples, {val_size} validation samples.")
    logger.info(f"Classes: {full_dataset.n_classes}")

    # ── Build model ──
    from src.models.baselines import BASELINE_REGISTRY
    from src.models.neurograph import NeuroGraphConformer

    logger.info("Building model...")

    n_classes = full_dataset.n_classes
    n_channels = cfg.dataset.get("n_channels_original", 64)
    n_samples = int(cfg.preprocessing.get("target_srate", 250) *
                    cfg.preprocessing.get("epoch_tmax", 2.0))

    if cfg.model.name in BASELINE_REGISTRY:
        model = BASELINE_REGISTRY[cfg.model.name](cfg.model)
    else:
        model = NeuroGraphConformer(
            cfg=cfg.model,
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=n_classes,
        )

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model parameters: {total_params:.2f}M")

    # ── Load pretrained (optional) ──
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        logger.info(f"Loaded pretrained weights from {args.pretrained}")

    # ── Train ──
    from src.training.trainer import Trainer

    logger.info("Starting training...")
    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        experiment_dir=exp_dir,
    )
    history = trainer.fit()

    # ── Save results ──
    with open(exp_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── Generate plots ──
    try:
        from src.evaluation.visualization import plot_training_curves, plot_confusion_matrix
        import numpy as np

        plot_training_curves(
            history,
            title=f"{args.experiment} — Training History",
            save_path=exp_dir / "training_curves.png",
        )
        logger.info(f"Training curves saved to {exp_dir / 'training_curves.png'}")

        # Confusion matrix from saved predictions
        npz_path = exp_dir / "val_predictions.npz"
        if npz_path.exists():
            data = np.load(npz_path)
            from sklearn.metrics import confusion_matrix as cm_func
            cm = cm_func(data["labels"], data["preds"])
            class_names = [str(i) for i in range(cm.shape[0])]
            plot_confusion_matrix(
                cm, class_names,
                title=f"{args.experiment} — Confusion Matrix",
                save_path=exp_dir / "confusion_matrix.png",
            )
            logger.info(f"Confusion matrix saved to {exp_dir / 'confusion_matrix.png'}")
    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")

    logger.info(f"Training complete. Results saved to {exp_dir}")


if __name__ == "__main__":
    main()

