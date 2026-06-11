from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from torch.utils.data import Dataset

from src.datasets.chisco import ChiscoDataset
from src.datasets.zuco import ZucoDataset
from src.datasets.thinking_out_loud import ThinkingOutLoudDataset

logger = logging.getLogger(__name__)

def get_dataset(
    dataset_name: str,
    manifest_path: str | Path,
    split: str = "train",
    subjects: list[str] | None = None,
    transform: Callable | None = None,
) -> Dataset:
    """Instantiate the appropriate PyTorch Dataset based on the dataset name.
    
    Args:
        dataset_name: The name of the dataset ('chisco', 'zuco', 'thinking_out_loud')
        manifest_path: Path to the trials.csv manifest
        split: Which split to load ('train', 'val', 'test')
        subjects: Optional list of subject IDs to restrict to
        transform: Optional torchvision-style transform callable
        
    Returns:
        A PyTorch Dataset ready for a DataLoader.
    """
    logger.info(f"Loading {dataset_name} dataset from {manifest_path} (split={split})")
    
    if dataset_name == "chisco":
        return ChiscoDataset(
            manifest_path=manifest_path,
            split=split,
            subjects=subjects,
            transform=transform,
        )
    elif dataset_name == "zuco":
        return ZucoDataset(
            manifest_path=manifest_path,
            split=split,
            subjects=subjects,
            transform=transform,
        )
    elif dataset_name == "thinking_out_loud":
        return ThinkingOutLoudDataset(
            manifest_path=manifest_path,
            split=split,
            subjects=subjects,
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
