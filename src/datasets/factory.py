from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from torch.utils.data import Dataset

from src.datasets.chisco import ChiscoDataset
from src.datasets.zuco import ZucoDataset
from src.datasets.thinking_out_loud import ThinkingOutLoudDataset

logger = logging.getLogger(__name__)

_REGISTRY = {
    "chisco": ChiscoDataset,
    "zuco": ZucoDataset,
    "thinking_out_loud": ThinkingOutLoudDataset,
}

def get_dataset(
    dataset_name: str,
    manifest_path: str | Path,
    split: str = "train",
    subjects: list[str] | None = None,
    transform: Callable | None = None,
    max_samples: int = 500,
) -> Dataset:
    """Instantiate the appropriate PyTorch Dataset based on the dataset name.
    
    Args:
        dataset_name: The name of the dataset ('chisco', 'zuco', 'thinking_out_loud')
        manifest_path: Path to the trials.csv manifest
        split: Which split to load ('train', 'val', 'test')
        subjects: Optional list of subject IDs to restrict to
        transform: Optional torchvision-style transform callable
        max_samples: Fixed number of time samples per trial (crop/pad)
        
    Returns:
        A PyTorch Dataset ready for a DataLoader.
    """
    if dataset_name == "Multiclass_Full_Run" or dataset_name == "all":
        from src.datasets.manifest import ManifestDataset
        logger.info(f"Loading ALL datasets for {dataset_name} (split={split}, max_samples={max_samples})")
        return ManifestDataset(
            manifest_path=manifest_path,
            dataset_name=dataset_name,
            split=split,
            subjects=subjects,
            transform=transform,
            max_samples=max_samples,
        )

    if dataset_name not in _REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(_REGISTRY.keys())} or 'Multiclass_Full_Run'")
    
    logger.info(f"Loading {dataset_name} (split={split}, max_samples={max_samples})")
    
    return _REGISTRY[dataset_name](
        manifest_path=manifest_path,
        split=split,
        subjects=subjects,
        transform=transform,
        max_samples=max_samples,
    )
