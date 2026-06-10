"""
profile_model.py — Profile inference latency and memory across model variants.

Usage:
    python scripts/profile_model.py --checkpoint results/checkpoints/best.pt \
        --variants small,medium,full --devices cpu,cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile model inference")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--variants", type=str, default="small,medium,full")
    parser.add_argument("--devices", type=str, default="cpu,cuda")
    parser.add_argument("--n-warmup", type=int, default=10)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--experiment", type=str, default="E8_deployment")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("profile", level="INFO")

    variants = args.variants.split(",")
    devices = args.devices.split(",")

    logger.info(f"Profiling variants: {variants}")
    logger.info(f"Devices: {devices}")
    logger.info(f"Warmup: {args.n_warmup}, Runs: {args.n_runs}")

    # TODO: For each variant x device:
    #   1. Build model
    #   2. Create dummy input
    #   3. Warmup
    #   4. Time inference
    #   5. Measure peak memory
    #   6. Count parameters and FLOPs
    #   7. Report percentiles (p50, p95, p99)

    logger.info("Profiling script ready. Implement after model architecture is finalized.")


if __name__ == "__main__":
    main()
