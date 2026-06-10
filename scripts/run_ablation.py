"""
run_ablation.py — Execute ablation study experiments.

Usage:
    python scripts/run_ablation.py --config configs/training/train_cls.yaml \
        --dataset kara_one --ablations A1,A2,A3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging import setup_logger

# Ablation definitions
ABLATIONS = {
    "A1": {
        "name": "Raw EEG vs Wavelet/Filter-bank",
        "description": "Compare raw EEG input vs CWT vs filter-bank",
        "config_overrides": [
            {"label": "raw", "overrides": ["model.frontend.type=raw"]},
            {"label": "cwt", "overrides": ["model.frontend.type=cwt"]},
            {"label": "filterbank", "overrides": ["model.frontend.type=filterbank"]},
        ],
    },
    "A2": {
        "name": "Without graph vs With graph",
        "description": "Remove graph spatial encoder",
        "config_overrides": [
            {"label": "no_graph", "overrides": ["model.spatial.type=none"]},
            {"label": "with_graph", "overrides": ["model.spatial.type=graph"]},
        ],
    },
    "A3": {
        "name": "Transformer vs Conformer",
        "description": "Replace Conformer with vanilla Transformer",
        "config_overrides": [
            {"label": "transformer", "overrides": ["model.encoder.type=transformer"]},
            {"label": "conformer", "overrides": ["model.encoder.type=conformer"]},
        ],
    },
    "A4": {
        "name": "Conformer vs Conformer+Mamba",
        "description": "Add/remove Mamba module",
        "config_overrides": [
            {"label": "no_mamba", "overrides": ["model.mamba.enabled=false"]},
            {"label": "with_mamba", "overrides": ["model.mamba.enabled=true"]},
        ],
    },
    "A5": {
        "name": "Without CTC vs With CTC",
        "description": "Remove CTC auxiliary loss",
        "config_overrides": [
            {"label": "no_ctc", "overrides": ["training.losses.ctc.enabled=false"]},
            {"label": "with_ctc", "overrides": ["training.losses.ctc.enabled=true"]},
        ],
    },
    "A6": {
        "name": "Without pretraining vs With pretraining",
        "description": "Skip self-supervised pretraining",
        "config_overrides": [
            {"label": "no_pretrain", "overrides": ["training.pretrained.enabled=false"]},
            {"label": "with_pretrain", "overrides": ["training.pretrained.enabled=true"]},
        ],
    },
    "A7": {
        "name": "Without RL vs SCST vs PPO",
        "description": "Compare RL methods",
        "config_overrides": [
            {"label": "no_rl", "overrides": ["model.rl.enabled=false"]},
            {"label": "scst", "overrides": ["model.rl.enabled=true", "model.rl.method=scst"]},
            {"label": "ppo", "overrides": ["model.rl.enabled=true", "model.rl.method=ppo"]},
        ],
    },
    "A8": {
        "name": "Channel strategy comparison",
        "description": "Common-channel vs Region-pooling vs Graph variable-channel",
        "config_overrides": [
            {"label": "common_channel", "overrides": ["features.channel_strategy=common_intersection"]},
            {"label": "region_pooling", "overrides": ["features.channel_strategy=region_pooling"]},
            {"label": "graph_variable", "overrides": ["features.channel_strategy=graph"]},
        ],
    },
    "A9": {
        "name": "Within-subject vs Cross-subject",
        "description": "Compare evaluation protocols",
        "config_overrides": [
            {"label": "within_subject", "overrides": ["protocol=within_subject"]},
            {"label": "cross_subject", "overrides": ["protocol=loso"]},
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ablations", type=str, default="A1,A2,A3,A4",
                        help="Comma-separated ablation IDs")
    parser.add_argument("--experiment", type=str, default="E7_ablation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("ablation", level="INFO")

    ablation_ids = args.ablations.split(",")
    logger.info(f"Running ablations: {ablation_ids}")

    for abl_id in ablation_ids:
        if abl_id not in ABLATIONS:
            logger.error(f"Unknown ablation: {abl_id}")
            continue

        abl = ABLATIONS[abl_id]
        logger.info(f"\n{'='*60}")
        logger.info(f"Ablation {abl_id}: {abl['name']}")
        logger.info(f"  {abl['description']}")
        logger.info(f"  Variants: {[v['label'] for v in abl['config_overrides']]}")

        for variant in abl["config_overrides"]:
            exp_name = f"{args.experiment}/{abl_id}/{variant['label']}"
            logger.info(f"\n  → Running variant: {variant['label']}")
            logger.info(f"    Overrides: {variant['overrides']}")
            logger.info(f"    Experiment: {exp_name}")

            # TODO: Call train.py with overrides
            # for seed in args.seeds:
            #     run_training(args.config, args.dataset, variant['overrides'],
            #                  experiment=f"{exp_name}/seed{seed}", seed=seed)

    logger.info("\nAblation suite configured. Implement training loop to execute.")


if __name__ == "__main__":
    main()
