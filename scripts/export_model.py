"""
export_model.py — Export model for deployment (ONNX/TorchScript).

Usage:
    python scripts/export_model.py --checkpoint results/checkpoints/best.pt \
        --format onnx --quantize int8 --output results/deployment/model.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export model for deployment")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "torchscript", "both"])
    parser.add_argument("--quantize", type=str, default="none", choices=["none", "int8", "dynamic"])
    parser.add_argument("--output", type=str, default="results/deployment/model")
    parser.add_argument("--n-channels", type=int, default=9, help="Input channels (region-pooled)")
    parser.add_argument("--n-samples", type=int, default=500, help="Input time samples")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("export", level="INFO")

    logger.info(f"Exporting model from: {args.checkpoint}")
    logger.info(f"Format: {args.format}, Quantize: {args.quantize}")

    # TODO: Load model, create dummy input, export
    logger.info("Export script ready. Implement after model training is complete.")


if __name__ == "__main__":
    main()
