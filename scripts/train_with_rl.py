"""
train_with_rl.py — Main training entry point with Stage 4 RL support.

Usage (Stage 3 - Supervised):
    python scripts/train_with_rl.py --config configs/models/conformer_small.yaml \
        --dataset chisco --protocol within_subject --experiment E1 --stage 3

Usage (Stage 4 - RL Fine-tuning):
    python scripts/train_with_rl.py --config configs/models/conformer_small.yaml \
        --dataset chisco --protocol within_subject --experiment E1 --stage 4 \
        --pretrained experiments/E1/best_model.pt
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
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config, merge_configs
from src.utils.logging import setup_logger
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NeuroGraph-Conformer Training (with RL support)")
    parser.add_argument("--config", type=str, required=True, help="Model config YAML")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["thinking_out_loud", "chisco", "zuco", "Multiclass_Full_Run"],
                        help="Dataset name")
    parser.add_argument("--protocol", type=str, default="within_subject",
                        choices=["within_subject", "loso", "cross_dataset"],
                        help="Evaluation protocol")
    parser.add_argument("--experiment", type=str, default="default",
                        help="Experiment name for logging")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained checkpoint")
    parser.add_argument("--stage", type=int, default=3,
                        choices=[3, 4],
                        help="Training stage: 3=supervised, 4=RL fine-tuning")
    parser.add_argument("--rl-config", type=str, default="configs/training/train_rl.yaml",
                        help="RL configuration file (used when --stage 4)")
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
    
    # If Stage 4 RL, load RL config
    if args.stage == 4:
        rl_cfg = load_config(args.rl_config)
        cfg = merge_configs(cfg, rl_cfg)
        logger.info(f"[STAGE 4] Loading RL config from {args.rl_config}")

    # Apply CLI overrides
    if args.overrides:
        from omegaconf import OmegaConf
        cli_cfg = OmegaConf.from_dotlist(args.overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    logger.info(f"Experiment: {args.experiment}")
    logger.info(f"Dataset: {args.dataset}, Protocol: {args.protocol}")
    logger.info(f"Model: {cfg.model.name}")
    logger.info(f"Stage: {args.stage}")

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

    # ── Initialize tokenizer ──
    from src.datasets.tokenizer import Tokenizer
    tokenizer = Tokenizer()
    vocab_size = tokenizer.vocab_size
    logger.info(f"Tokenizer initialized: vocab_size={vocab_size}")

    # ── Build dataset (with tokenizer for text targets) ──
    from src.datasets.factory import get_dataset
    from src.datasets.manifest import collate_fn

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
        tokenizer=tokenizer,
    )

    if len(full_dataset) == 0:
        logger.error(f"No samples found for {args.dataset} in {manifest_path}")
        return

    # ── Stratified split (avoids rare-class crash from random_split) ──
    n_classes = full_dataset.n_classes
    logger.info(f"Classes: {n_classes}")

    if n_classes < 2:
        logger.warning(
            f"Only {n_classes} class(es) detected! Training will be degenerate. "
            f"Check label extraction in build_manifests.py."
        )

    # Try stratified split, fallback to random
    try:
        from sklearn.model_selection import StratifiedShuffleSplit

        labels_array = []
        for i in range(len(full_dataset)):
            row = full_dataset.dataset.df.iloc[i] if hasattr(full_dataset, 'dataset') else full_dataset.df.iloc[i]
            label_col = full_dataset.dataset._label_col if hasattr(full_dataset, 'dataset') else full_dataset._label_col
            if label_col and label_col in row.index:
                labels_array.append(str(row[label_col]))
            else:
                labels_array.append("0")

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
        train_idx, valtest_idx = next(splitter.split(range(len(full_dataset)), labels_array))

        valtest_labels = [labels_array[i] for i in valtest_idx]
        splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=args.seed)
        val_rel_idx, test_rel_idx = next(splitter2.split(range(len(valtest_idx)), valtest_labels))
        val_idx = valtest_idx[val_rel_idx]
        test_idx = valtest_idx[test_rel_idx]

        train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
        test_dataset = torch.utils.data.Subset(full_dataset, test_idx)

        logger.info(f"Stratified split: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    except Exception as e:
        logger.warning(f"Stratified split failed ({e}), using random split.")
        test_size = int(0.1 * len(full_dataset))
        val_size = int(0.1 * len(full_dataset))
        train_size = len(full_dataset) - val_size - test_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(args.seed)
        )

    # ── DataLoaders with custom collate_fn ──
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

    # ── Compute class weights ──
    class_weights = None
    if hasattr(full_dataset, 'get_class_weights'):
        class_weights = full_dataset.get_class_weights()
        logger.info(f"Class weights computed: {class_weights.tolist()[:5]}...")

    # ── Build model (with vocab_size for generation head) ──
    from src.models.baselines import BASELINE_REGISTRY
    from src.models.neurograph import NeuroGraphConformer

    logger.info("Building model...")

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

    # ── Load pretrained (optional) ──
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        logger.info(f"Loaded pretrained weights from {args.pretrained}")

    # ── Compute adjacency matrix for graph encoder ──
    adjacency = None
    try:
        from src.features.adjacency import compute_adjacency_matrix
        adjacency = compute_adjacency_matrix(
            n_channels=n_channels,
            method=cfg.model.get("arch", {}).get("spatial", {}).get("adjacency", "distance"),
        )
        logger.info(f"Adjacency matrix: {adjacency.shape}")
    except Exception as e:
        logger.warning(f"Could not compute adjacency matrix: {e}. Using default.")

    # ── Train (Stage 3: Supervised) ──
    from src.training.trainer import Trainer
    from src.training.wandb_logger import create_wandb_logger

    logger.info(f"Starting Stage {args.stage} training...")
    
    # Initialize W&B logger if enabled
    wb_logger = None
    if cfg.logging.get("wandb", {}).get("enabled", False):
        logger.info("Initializing W&B logger...")
        from src.utils.config import config_to_dict
        wb_logger = create_wandb_logger(
            cfg=config_to_dict(cfg),
            experiment_name=f"{args.experiment}_stage{args.stage}",
            tags=[args.dataset, cfg.model.name, f"stage_{args.stage}"]
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
    if hasattr(trainer, "_last_test_preds"):
        np.savez(
            exp_dir / "test_predictions.npz",
            preds=trainer._last_test_preds,
            labels=trainer._last_test_labels,
        )
        logger.info(f"Saved test predictions to {exp_dir / 'test_predictions.npz'}")
        
    history["test_metrics"] = test_metrics
    
    # ── Stage 4: RL fine-tuning ──
    if args.stage == 4:
        logger.info("="*70)
        logger.info("[STAGE 4/4] RL Fine-tuning with SCST")
        logger.info("="*70)
        
        # Check if pretrained model was saved from Stage 3
        stage3_ckpt = args.pretrained or exp_dir / "checkpoints" / "best.pt"
        if isinstance(stage3_ckpt, str):
            stage3_ckpt = Path(stage3_ckpt)
        
        if not stage3_ckpt.exists():
            logger.warning(f"No Stage 3 checkpoint found at {stage3_ckpt}. Skipping RL fine-tuning.")
        elif model.generation_head is None:
            logger.error(
                "Cannot run RL fine-tuning: model has no generation_head. "
                "Ensure the model config includes heads.generation and vocab_size is set."
            )
        else:
            logger.info(f"Loading Stage 3 checkpoint: {stage3_ckpt}")
            checkpoint = torch.load(stage3_ckpt, map_location=args.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            
            # Initialize RL trainer
            from src.training.rl_trainer import RLTrainer
            from src.utils.device import DeviceManager

            device_mgr = DeviceManager(args.device)
            device = device_mgr.device
            model = model.to(device)
            
            # Get RL config from merged config (training.rl or model.rl)
            rl_cfg = cfg.get("training", {}).get("rl", cfg.get("model", {}).get("rl", {}))
            
            rl_trainer = RLTrainer(
                model=model,
                cfg=rl_cfg,
                device=device,
            )
            logger.info(f"RL Trainer initialized with method={rl_cfg.get('method', 'scst')}")
            
            # RL training loop
            rl_epochs = rl_cfg.get("max_epochs", cfg.training.get("max_epochs", 10))
            rl_lr = rl_cfg.get("lr", 1e-5)
            rl_optimizer = torch.optim.AdamW(
                model.parameters(), lr=rl_lr, weight_decay=1e-4
            )
            
            rl_history = {"rl_loss": [], "greedy_reward": [], "sample_reward": [], "advantage": []}
            best_reward = -float("inf")
            
            logger.info(f"Starting RL training for {rl_epochs} epochs (lr={rl_lr})")
            
            for epoch in range(rl_epochs):
                model.train()
                epoch_metrics = {"rl_loss": 0.0, "greedy_reward": 0.0, "sample_reward": 0.0, "advantage": 0.0}
                n_batches = 0
                
                from tqdm import tqdm
                pbar = tqdm(train_loader, desc=f"RL Epoch {epoch+1}/{rl_epochs}", leave=False)
                
                for batch in pbar:
                    eeg = batch["eeg"].to(device)
                    texts = batch["text"]
                    
                    # Skip batches without text
                    target_lengths = batch.get("target_length", torch.tensor(0)).to(device)
                    if target_lengths.sum() == 0:
                        continue
                    
                    # Forward encoder
                    adj = None
                    if adjacency is not None:
                        adj = adjacency.to(device)
                        if adj.shape[0] != eeg.shape[1]:
                            b_c = eeg.shape[1]
                            min_c = min(adj.shape[0], b_c)
                            new_adj = torch.eye(b_c, device=device)
                            new_adj[:min_c, :min_c] = adj[:min_c, :min_c]
                            adj = new_adj
                    
                    encoder_output = model.encode(eeg, adj=adj)
                    
                    # RL step
                    rl_metrics = rl_trainer.train_step(encoder_output, texts, tokenizer)
                    
                    if "rl_loss_tensor" in rl_metrics:
                        rl_loss = rl_metrics["rl_loss_tensor"]
                        rl_optimizer.zero_grad()
                        rl_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        rl_optimizer.step()
                    
                    for k in epoch_metrics:
                        epoch_metrics[k] += rl_metrics.get(k, rl_metrics.get("rl_loss", 0.0)) if k == "rl_loss" else rl_metrics.get(k, 0.0)
                    n_batches += 1
                    
                    pbar.set_postfix({
                        "rl_loss": f"{rl_metrics.get('rl_loss', 0):.4f}",
                        "reward": f"{rl_metrics.get('sample_reward', 0):.4f}",
                    })
                
                if n_batches > 0:
                    for k in epoch_metrics:
                        epoch_metrics[k] /= n_batches
                        rl_history[k].append(epoch_metrics[k])
                    
                    logger.info(
                        f"RL Epoch {epoch+1}/{rl_epochs} | "
                        f"rl_loss={epoch_metrics['rl_loss']:.4f} | "
                        f"greedy_reward={epoch_metrics['greedy_reward']:.4f} | "
                        f"sample_reward={epoch_metrics['sample_reward']:.4f} | "
                        f"advantage={epoch_metrics['advantage']:.4f}"
                    )
                    
                    # Save best RL checkpoint
                    if epoch_metrics["sample_reward"] > best_reward:
                        best_reward = epoch_metrics["sample_reward"]
                        ckpt_dir = exp_dir / "checkpoints"
                        ckpt_dir.mkdir(parents=True, exist_ok=True)
                        torch.save({
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "rl_metrics": epoch_metrics,
                        }, ckpt_dir / "best_rl.pt")
                        logger.info(f"  ✓ New best RL reward: {best_reward:.4f}")
                else:
                    logger.warning(f"RL Epoch {epoch+1}: no batches had text data.")
            
            history["rl_history"] = rl_history
            logger.info(f"RL fine-tuning complete. Best reward: {best_reward:.4f}")

    # ── Save results ──
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
        from src.evaluation.visualization import plot_training_curves, plot_confusion_matrix

        plot_training_curves(
            history,
            title=f"{args.experiment} — Training History (Stage {args.stage})",
            save_path=exp_dir / "training_curves.png",
        )
        logger.info(f"Training curves saved to {exp_dir / 'training_curves.png'}")

        # Confusion matrix from saved predictions
        for split in ["val", "test"]:
            npz_path = exp_dir / f"{split}_predictions.npz"
            if npz_path.exists():
                data = np.load(npz_path)
                from sklearn.metrics import confusion_matrix as cm_func
                cm = cm_func(data["labels"], data["preds"])
                class_names = [str(i) for i in range(cm.shape[0])]
                plot_confusion_matrix(
                    cm, class_names,
                    title=f"{args.experiment} — Confusion Matrix ({split}) (Stage {args.stage})",
                    save_path=exp_dir / f"confusion_matrix_{split}.png",
                )
                logger.info(f"Confusion matrix ({split}) saved to {exp_dir / f'confusion_matrix_{split}.png'}")
    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")

    logger.info(f"Stage {args.stage} training complete. Results saved to {exp_dir}")


if __name__ == "__main__":
    main()
