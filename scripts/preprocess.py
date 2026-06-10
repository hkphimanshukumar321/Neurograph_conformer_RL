"""
preprocess.py — Run preprocessing pipeline on a dataset.

Usage:
    python scripts/preprocess.py --config configs/datasets/kara_one.yaml \
        --pipeline unified --output data/processed/kara_one
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config, merge_configs
from src.utils.logging import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EEG Preprocessing Pipeline")
    parser.add_argument("--config", type=str, required=True,
                        help="Dataset config YAML")
    parser.add_argument("--pipeline", type=str, default="unified",
                        choices=["unified", "provider"],
                        help="Preprocessing pipeline mode")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: from config)")
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="Specific subject IDs to process")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Number of parallel jobs")
    return parser.parse_args()


def main():
    args = parse_args()

    logger = setup_logger("preprocess", level="INFO")

    # Load configs
    base_cfg = load_config("configs/base.yaml")
    dataset_cfg = load_config(args.config)
    cfg = merge_configs(base_cfg, dataset_cfg)

    dataset_name = cfg.dataset.name
    logger.info(f"Preprocessing dataset: {dataset_name}")
    logger.info(f"Pipeline mode: {args.pipeline}")

    # Determine output directory
    output_dir = Path(args.output) if args.output else Path(cfg.dataset.processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get preprocessing config
    if args.pipeline == "provider":
        preproc_cfg = cfg.dataset.get("provider_preprocessing", cfg.preprocessing)
        logger.info("Using provider-matched preprocessing settings")
    else:
        preproc_cfg = cfg.preprocessing
        logger.info("Using unified preprocessing settings")

    logger.info(f"Target sampling rate: {preproc_cfg.get('target_srate', cfg.preprocessing.target_srate)} Hz")
    logger.info(f"Bandpass: {preproc_cfg.get('bandpass_low', cfg.preprocessing.bandpass_low)}"
                f"–{preproc_cfg.get('bandpass_high', cfg.preprocessing.bandpass_high)} Hz")
    logger.info(f"Artifact method: {preproc_cfg.get('artifact_method', cfg.preprocessing.artifact_method)}")

    # TODO: Implement dataset-specific raw data loading
    # For each subject:
    #   1. Load raw EEG (MNE)
    #   2. Run PreprocessingPipeline
    #   3. Save preprocessed epochs to output_dir
    #   4. Generate quality report

    logger.info(
        f"Preprocessing pipeline configured for {dataset_name}. "
        "Implement raw data loading for your specific dataset format."
    )
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
