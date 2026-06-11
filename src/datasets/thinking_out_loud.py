from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.datasets.manifest import ManifestDataset

class ThinkingOutLoudDataset(ManifestDataset):
    """Dataset loader for the Thinking Out Loud dataset."""
    
    def __init__(
        self,
        manifest_path: str | Path,
        split: str = "train",
        subjects: list[str] | None = None,
        transform: Callable | None = None,
        max_samples: int = 500,
    ):
        super().__init__(
            manifest_path=manifest_path,
            dataset_name="thinking_out_loud",
            split=split,
            subjects=subjects,
            transform=transform,
            max_samples=max_samples,
        )
