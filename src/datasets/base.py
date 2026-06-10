"""
Base dataset class — abstract interface for all EEG datasets.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class BaseEEGDataset(Dataset, ABC):
    """Abstract base class for EEG datasets.

    All dataset implementations must inherit from this class and implement
    the abstract methods.

    Parameters
    ----------
    data_dir : str or Path
        Path to preprocessed data directory.
    split : str
        Data split: 'train', 'val', 'test'.
    subjects : list[str], optional
        List of subject IDs to include. None = all subjects.
    transform : callable, optional
        Optional transform applied to each sample.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        subjects: list[str] | None = None,
        transform: Any | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.subjects = subjects
        self.transform = transform

        # Load data
        self.epochs, self.labels, self.metadata = self._load_data()

    @abstractmethod
    def _load_data(self) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        """Load preprocessed epochs, labels, and per-trial metadata.

        Returns
        -------
        tuple of:
            epochs : np.ndarray, shape (N_trials, C, T)
            labels : np.ndarray, shape (N_trials,)
            metadata : list[dict], length N_trials
        """
        ...

    @abstractmethod
    def get_label_map(self) -> dict[int, str]:
        """Return mapping from label ID to label name."""
        ...

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Return the dataset name (e.g., 'kara_one')."""
        ...

    @property
    @abstractmethod
    def n_classes(self) -> int:
        """Return number of classes."""
        ...

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample.

        Returns
        -------
        dict with keys:
            'eeg': torch.Tensor, shape (C, T)
            'label': torch.Tensor, scalar
            'subject_id': str
            'trial_id': str
            'dataset': str
        """
        eeg = self.epochs[idx].astype(np.float32)
        label = int(self.labels[idx])

        if self.transform is not None:
            eeg = self.transform(eeg)

        sample = {
            "eeg": torch.from_numpy(eeg),
            "label": torch.tensor(label, dtype=torch.long),
            "dataset": self.dataset_name,
        }

        # Add metadata if available
        if self.metadata and idx < len(self.metadata):
            meta = self.metadata[idx]
            sample["subject_id"] = meta.get("subject_id", "unknown")
            sample["trial_id"] = meta.get("trial_id", f"trial_{idx:05d}")
        else:
            sample["subject_id"] = "unknown"
            sample["trial_id"] = f"trial_{idx:05d}"

        return sample

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for balanced training."""
        unique, counts = np.unique(self.labels, return_counts=True)
        total = len(self.labels)
        weights = total / (len(unique) * counts)
        weight_tensor = torch.zeros(self.n_classes)
        for cls, w in zip(unique, weights):
            weight_tensor[int(cls)] = w
        return weight_tensor

    def get_subject_ids(self) -> list[str]:
        """Return list of unique subject IDs in this split."""
        if self.metadata:
            return list(set(m.get("subject_id", "unknown") for m in self.metadata))
        return ["unknown"]
