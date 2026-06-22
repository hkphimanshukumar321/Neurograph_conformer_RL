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
    import os
    dataset_cfg_path = f"configs/datasets/{args.dataset}.yaml"
    if os.path.exists(dataset_cfg_path):
        dataset_cfg = load_config(dataset_cfg_path)
    else:
        logger.warning(f"Dataset config {dataset_cfg_path} not found. Using default empty dataset config.")
        from omegaconf import OmegaConf
        dataset_cfg = OmegaConf.create({"dataset": {"name": args.dataset}})
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

    # ── Split Dataset (Stratified) ──
    # Use stratified splitting to ensure each class appears in train/val/test
    from sklearn.model_selection import StratifiedShuffleSplit
    import numpy as np

    all_labels_for_split = []
    for i in range(len(full_dataset)):
        row = full_dataset.df.iloc[i]
        label_col = full_dataset._label_col
        if label_col and label_col in row.index:
            raw_lab = str(row[label_col])
            all_labels_for_split.append(full_dataset.label_str_to_int.get(raw_lab, 0))
        else:
            all_labels_for_split.append(0)
    all_labels_for_split = np.array(all_labels_for_split)

    # First split: separate test set (10%)
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=args.seed)
    train_val_idx, test_idx = next(sss_test.split(np.zeros(len(all_labels_for_split)), all_labels_for_split))

    # Second split: separate val from train (10% of original = ~11% of train_val)
    val_frac = 0.1 / 0.9  # adjust fraction since we're splitting from 90%
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=args.seed)
    train_idx, val_idx = next(sss_val.split(
        np.zeros(len(train_val_idx)), all_labels_for_split[train_val_idx]
    ))
    # Map back to original indices
    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]

    train_dataset = torch.utils.data.Subset(full_dataset, train_idx.tolist())
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx.tolist())
    test_dataset = torch.utils.data.Subset(full_dataset, test_idx.tolist())

    train_size = len(train_dataset)
    val_size = len(val_dataset)
    test_size = len(test_dataset)

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
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    logger.info(f"Loaded {train_size} train, {val_size} val, {test_size} test samples.")
    logger.info(f"Classes: {full_dataset.n_classes}")
    # Log class distribution
    if hasattr(full_dataset, 'label_int_to_str') and hasattr(full_dataset, '_label_col'):
        label_col = full_dataset._label_col
        if label_col and label_col in full_dataset.df.columns:
            counts = full_dataset.df[label_col].value_counts()
            logger.info(f"Class distribution:")
            for lab, count in counts.items():
                idx = full_dataset.label_str_to_int.get(str(lab), '?')
                logger.info(f"  [{idx}] {lab}: {count} samples ({100*count/len(full_dataset):.1f}%)")

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

    try:
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(f"Model parameters: {total_params:.2f}M")
    except ValueError:
        logger.info("Model parameters: (deferred until first forward pass due to lazy modules)")
    
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

    # ── Compute class weights for balanced training ──
    class_weights = None
    if hasattr(full_dataset, 'get_class_weights'):
        class_weights = full_dataset.get_class_weights()
        logger.info(f"Class weights computed: {class_weights.tolist()[:5]}{'...' if len(class_weights) > 5 else ''}")

    # ── Compute adjacency matrix from electrode positions ──
    adjacency = None
    spatial_type = cfg.model.get("spatial", cfg.model.get("arch", {}).get("spatial", {})).get("type", "none")
    if spatial_type == "graph":
        from src.utils.electrode_positions import compute_adjacency_for_batch
        adjacency = compute_adjacency_for_batch(n_channels, device="cpu")
        logger.info(f"Pre-computed adjacency matrix: shape={adjacency.shape}")

    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        experiment_dir=exp_dir,
        logger_obj=wb_logger,
        tokenizer=tokenizer,
        class_weights=class_weights,
        adjacency=adjacency,
    )
    history = trainer.fit()
    
    if wb_logger is not None:
        wb_logger.finish()

    # ── Evaluate on Test Set ──
    logger.info("Loading best model for test set evaluation...")
    trainer.load_checkpoint(exp_dir / "checkpoints" / "best.pt")
    test_metrics = trainer.validate(loader=test_loader, prefix="test")
    logger.info(f"Test metrics: {test_metrics}")
    
    # Save test predictions
    import numpy as np
    if hasattr(trainer, "_last_test_preds"):
        np.savez(
            exp_dir / "test_predictions.npz",
            preds=trainer._last_test_preds,
            labels=trainer._last_test_labels,
        )
        logger.info(f"Saved test predictions to {exp_dir / 'test_predictions.npz'}")
        
    history["test_metrics"] = test_metrics

    # ── Save results ──
    import numpy as np
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            return super().default(obj)

    with open(exp_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2, cls=NumpyEncoder)

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
        for split in ["val", "test"]:
            npz_path = exp_dir / f"{split}_predictions.npz"
            if npz_path.exists():
                data = np.load(npz_path)
                from sklearn.metrics import confusion_matrix as cm_func
                cm = cm_func(data["labels"], data["preds"])
                class_names = [
                    full_dataset.label_int_to_str.get(i, str(i))
                    for i in range(cm.shape[0])
                ] if hasattr(full_dataset, 'label_int_to_str') else [str(i) for i in range(cm.shape[0])]
                plot_confusion_matrix(
                    cm, class_names,
                    title=f"{args.experiment} — Confusion Matrix ({split})",
                    save_path=exp_dir / f"confusion_matrix_{split}.png",
                )
                logger.info(f"Confusion matrix ({split}) saved to {exp_dir / f'confusion_matrix_{split}.png'}")
    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")

    logger.info(f"Training complete. Results saved to {exp_dir}")


if __name__ == "__main__":
    main()

