"""
evaluate.py — Evaluation entry point.

Usage:
    python scripts/evaluate.py --checkpoint results/checkpoints/best.pt \
        --dataset kara_one --protocol loso --metrics all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--protocol", type=str, default="loso")
    parser.add_argument("--metrics", type=str, default="all")
    parser.add_argument("--output", type=str, default="results/evaluation")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("evaluate", level="INFO")

    logger.info(f"Evaluating checkpoint: {args.checkpoint}")
    logger.info(f"Dataset: {args.dataset}, Protocol: {args.protocol}")

    # TODO: Load model, run evaluation, save results
    logger.info("Evaluation script ready. Implement after training pipeline is complete.")


if __name__ == "__main__":
    main()
