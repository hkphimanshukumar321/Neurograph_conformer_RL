from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from src.datasets.manifest import ManifestDataset
from src.datasets.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class ChiscoDataset(ManifestDataset):
    """Dataset loader for the Chisco (Chinese Imagined Speech) dataset.

    Extends ManifestDataset with Chisco-specific validation:
    - Warns if only 1 class is detected (likely label extraction failure)
    - Validates that text field is populated for generation tasks
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str = "train",
        subjects: list[str] | None = None,
        transform: Callable | None = None,
        max_samples: int = 500,
        tokenizer: Tokenizer | None = None,
    ):
        super().__init__(
            manifest_path=manifest_path,
            dataset_name="chisco",
            split=split,
            subjects=subjects,
            transform=transform,
            max_samples=max_samples,
            tokenizer=tokenizer,
        )

        # ── Chisco-specific validation ──
        if self.n_classes < 2:
            logger.warning(
                f"ChiscoDataset: Only {self.n_classes} class(es) detected! "
                f"Chisco should have multiple classes (run numbers map to sentence stimuli). "
                f"Check that build_manifests.py correctly populated the 'label' column in trials.csv. "
                f"Label column used: '{self._label_col}'"
            )
            if self._label_col and self._label_col in self.df.columns:
                sample_labels = self.df[self._label_col].head(10).tolist()
                logger.warning(f"  Sample labels: {sample_labels}")

        # Validate text field for generation tasks
        if "text" in self.df.columns:
            n_with_text = self.df["text"].apply(
                lambda x: isinstance(x, str) and len(x.strip()) > 0
            ).sum()
            pct = 100.0 * n_with_text / max(len(self.df), 1)
            if pct < 50:
                logger.warning(
                    f"ChiscoDataset: Only {pct:.0f}% of trials have non-empty text. "
                    f"Generation/RL training will be limited."
                )
            else:
                logger.info(
                    f"ChiscoDataset: {pct:.0f}% of trials have text labels "
                    f"({n_with_text}/{len(self.df)})"
                )
