"""
scan_datasets.py — Scan raw data directories and generate metadata manifests.

Usage:
    python scripts/scan_datasets.py --data-dir data/raw --output data/manifests
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.io import save_json, save_tsv
from src.utils.logging import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan datasets and generate manifests")
    parser.add_argument("--data-dir", type=str, default="data/raw",
                        help="Root directory of raw datasets")
    parser.add_argument("--output", type=str, default="data/manifests",
                        help="Output directory for manifests")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("scan", level="INFO")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Scanning datasets in: {data_dir}")

    # Check which datasets exist
    datasets_found = []
    for dataset_name in ["kara_one", "thinking_out_loud", "chisco", "zuco"]:
        dataset_path = data_dir / dataset_name
        if dataset_path.exists():
            datasets_found.append(dataset_name)
            logger.info(f"  ✓ Found: {dataset_name}")
        else:
            logger.warning(f"  ✗ Not found: {dataset_name} (expected at {dataset_path})")

    if not datasets_found:
        logger.error("No datasets found! Please download datasets to data/raw/")
        logger.info("Expected structure:")
        logger.info("  data/raw/kara_one/")
        logger.info("  data/raw/thinking_out_loud/")
        logger.info("  data/raw/chisco/")
        logger.info("  data/raw/zuco/")
        return

    # Generate manifest skeleton
    manifest = {
        "datasets": [],
        "generated_by": "scan_datasets.py",
    }

    for name in datasets_found:
        manifest["datasets"].append({
            "name": name,
            "raw_dir": str(data_dir / name),
            "status": "found",
            "n_subjects": "unknown",
            "n_files": len(list((data_dir / name).rglob("*"))),
        })

    save_json(manifest, output_dir / "dataset_manifest.json")
    logger.info(f"Manifest saved to: {output_dir / 'dataset_manifest.json'}")
    logger.info(f"Found {len(datasets_found)} datasets. Run preprocessing next.")


if __name__ == "__main__":
    main()
