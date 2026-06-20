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

# Globally suppress annoying sklearn warnings about unique classes
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.*")
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

    # ── Compute fixed dimensions from config ──
    n_channels = cfg.dataset.get("n_channels_original", 64)
    n_samples = int(cfg.preprocessing.get("target_srate", 250) *
                    cfg.preprocessing.get("epoch_tmax", 2.0))

    # ── Build dataset ──
    from src.datasets.factory import get_dataset

    logger.info(f"Loading {args.dataset} dataset (n_samples={n_samples})...")
    project_root = Path(__file__).resolve().parent.parent
    manifest_path = project_root / "data" / "processed" / "manifests" / "trials.csv"

    if not manifest_path.exists():
        logger.error(f"Manifest not found at {manifest_path}. Run preprocessing first.")
        return

    full_dataset = get_dataset(
        args.dataset,
        manifest_path=manifest_path,
        split="train",
        max_samples=n_samples,
    )

    if len(full_dataset) == 0:
        logger.error(f"No samples found for {args.dataset} in {manifest_path}")
        return

    # ── Tokenizer ──
    from src.datasets.tokenizer import Tokenizer
    tokenizer = None
    vocab_size = None

    decoder_cfg = cfg.model.get("decoder", cfg.model.get("arch", {}).get("decoder", {}))
    if decoder_cfg.get("enabled", False):
        gen_cfg = cfg.model.get("heads", cfg.model.get("arch", {}).get("heads", {})).get("generation", {})
        if cfg.training.get("loss_weights", {}).get("gen", 0) > 0 or cfg.training.get("loss_weights", {}).get("rl", 0) > 0:
            logger.info("Initializing Tokenizer for Generation/RL...")
            tokenizer_loaded = False
            if args.pretrained:
                pretrained_dir = Path(args.pretrained).parent
                # if pretrained is just "best.pt", parent is "."
                # check if tokenizer.json is near it or in its parent (experiments/phase2)
                tok_path = pretrained_dir / "tokenizer.json"
                if not tok_path.exists():
                    # sometimes pretrained is experiments/exp_name/checkpoints/best.pt
                    tok_path = pretrained_dir.parent / "tokenizer.json"
                    
                if tok_path.exists():
                    logger.info(f"Loading tokenizer from {tok_path}")
                    tokenizer = Tokenizer.load(tok_path)
                    vocab_size = tokenizer.vocab_size
                    tokenizer_loaded = True
            
            if not tokenizer_loaded:
                tokenizer = Tokenizer(mode="char" if args.dataset == "chisco" else "word")
                texts = [item["text"] for item in full_dataset.df.to_dict("records")]
                tokenizer.fit(texts)
                vocab_size = tokenizer.vocab_size
                logger.info(f"Tokenizer fitted with vocab_size={vocab_size}")
                
            tokenizer.save(exp_dir / "tokenizer.json")
            
            # Update dataset to use tokenizer
            full_dataset.tokenizer = tokenizer
        else:
            logger.info("Generation head config found but generation/RL weights are 0 — skipping generation head (Phase 1)")

    # ── Split Dataset ──
    # ... previous split code ...
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    from src.datasets.manifest import collate_fn
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    logger.info(f"Loaded {train_size} training samples, {val_size} validation samples.")
    logger.info(f"Classes: {full_dataset.n_classes}")

    # ── Build model ──
    from src.models.baselines import BASELINE_REGISTRY
    from src.models.neurograph import NeuroGraphConformer

    logger.info("Building model...")

    n_classes = full_dataset.n_classes

    if cfg.model.name in BASELINE_REGISTRY:
        model = BASELINE_REGISTRY[cfg.model.name](cfg.model)
    else:
        model = NeuroGraphConformer(
            cfg=cfg.model,
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=n_classes,
            vocab_size=vocab_size,
        )

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model parameters: {total_params:.2f}M")
    
    # Log active heads
    if hasattr(model, 'classification_heads'):
        logger.info(f"  Classification heads: {list(model.classification_heads.keys())}")
    if hasattr(model, 'retrieval_head') and model.retrieval_head is not None:
        logger.info(f"  Retrieval head: ACTIVE")
    if hasattr(model, 'generation_head') and model.generation_head is not None:
        logger.info(f"  Generation head: ACTIVE")
    else:
        logger.info(f"  Generation head: INACTIVE (Phase 1)")

    # ── Load pretrained (optional) ──
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        logger.info(f"Loaded pretrained weights from {args.pretrained}")

    # ── Train ──
    from src.training.trainer import Trainer
    from src.training.wandb_logger import create_wandb_logger

    logger.info("Starting training...")
    
    # Initialize W&B logger if enabled
    wb_logger = None
    if cfg.logging.get("wandb", {}).get("enabled", False):
        logger.info("Initializing W&B logger...")
        from src.utils.config import config_to_dict
        wb_logger = create_wandb_logger(
            cfg=config_to_dict(cfg),
            experiment_name=args.experiment,
            tags=[args.dataset, cfg.model.name]
        )

    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        experiment_dir=exp_dir,
        logger_obj=wb_logger,
        tokenizer=tokenizer,
    )
    history = trainer.fit()
    
    if wb_logger is not None:
        wb_logger.finish()

    # ── Save results ──
    with open(exp_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── Generate plots ──
    try:
        from src.evaluation.visualization import (
            plot_training_curves,
            plot_confusion_matrix,
            plot_multitask_losses,
            plot_reward_curves,
        )
        import numpy as np

        # 1. Standard training curves (loss + accuracy)
        plot_training_curves(
            history,
            title=f"{args.experiment} — Training History",
            save_path=exp_dir / "training_curves.png",
        )
        logger.info(f"Training curves saved to {exp_dir / 'training_curves.png'}")

        # 2. Multi-task loss breakdown (cls vs contrastive vs total)
        plot_multitask_losses(
            history,
            title=f"{args.experiment} — Multi-Task Loss Breakdown",
            save_path=exp_dir / "multitask_losses.png",
        )
        logger.info(f"Multi-task loss plot saved to {exp_dir / 'multitask_losses.png'}")

        # 3. RL reward curves (only if RL was active)
        if any(len(history.get(k, [])) > 0 for k in ["rl_greedy_reward", "rl_sample_reward"]):
            plot_reward_curves(
                history,
                title=f"{args.experiment} — RL Reward Progress",
                save_path=exp_dir / "rl_reward_curves.png",
            )
            logger.info(f"RL reward curves saved to {exp_dir / 'rl_reward_curves.png'}")

        # 4. Confusion matrix from saved predictions
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

