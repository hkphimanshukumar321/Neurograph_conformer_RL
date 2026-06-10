"""
train.py — Main training entry point.

Usage:
    python scripts/train.py --config configs/models/conformer_medium.yaml \
        --dataset kara_one --protocol loso --experiment E1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config, merge_configs
from src.utils.logging import setup_logger
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NeuroGraph-Conformer Training")
    parser.add_argument("--config", type=str, required=True, help="Model config YAML")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["kara_one", "thinking_out_loud", "chisco", "zuco"],
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
    logger.info("Loading dataset...")
    # TODO: Implement dataset loading based on args.dataset and args.protocol
    # train_dataset = get_dataset(args.dataset, split="train", cfg=cfg)
    # val_dataset = get_dataset(args.dataset, split="val", cfg=cfg)
    # train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, ...)
    # val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, ...)

    logger.info(
        "Dataset loading not yet implemented. "
        "Run preprocessing first: python scripts/preprocess.py"
    )

    # ── Build model ──
    # model = build_model(cfg)
    # logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # ── Load pretrained (optional) ──
    # if args.pretrained:
    #     checkpoint = torch.load(args.pretrained)
    #     model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    #     logger.info(f"Loaded pretrained weights from {args.pretrained}")

    # ── Train ──
    # trainer = Trainer(model, cfg, train_loader, val_loader, device=args.device, experiment_dir=exp_dir)
    # history = trainer.fit()

    # ── Save results ──
    # save_json(history, exp_dir / "history.json")

    logger.info("Training script initialized successfully. Implement dataset loading to proceed.")


if __name__ == "__main__":
    main()
