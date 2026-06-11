from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.datasets.manifest import ManifestDataset

class ChiscoDataset(ManifestDataset):
    """Dataset loader for the Chisco dataset."""
    
    def __init__(
        self,
        manifest_path: str | Path,
        split: str = "train",
        subjects: list[str] | None = None,
        transform: Callable | None = None,
    ):
        super().__init__(
            manifest_path=manifest_path,
            dataset_name="chisco",
            split=split,
            subjects=subjects,
            transform=transform,
        )
