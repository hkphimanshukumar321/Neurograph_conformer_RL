"""
build_splits.py — Generate train/val/test splits with leakage prevention.

Usage:
    python scripts/build_splits.py --manifest data/manifests/dataset_manifest.json \
        --protocols within_subject,loso --output data/splits
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build data splits")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--protocols", type=str, default="within_subject,loso")
    parser.add_argument("--output", type=str, default="data/splits")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("splits", level="INFO")

    protocols = args.protocols.split(",")
    logger.info(f"Building splits: protocols={protocols}, n_folds={args.n_folds}")

    # TODO: Implement split generation with:
    #   - Subject-level splitting (no subject leakage)
    #   - Session-aware splits (no session leakage)
    #   - Stratified by label distribution
    #   - LOSO (leave-one-subject-out)
    #   - Within-subject k-fold

    logger.info("Split generation ready. Implement after preprocessing is complete.")


if __name__ == "__main__":
    main()
