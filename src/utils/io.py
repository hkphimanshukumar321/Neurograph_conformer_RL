"""
Data I/O utilities — loading/saving EEG data, manifests, splits, and results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ──── JSON I/O ────


def load_json(path: str | Path) -> dict | list:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict | list, path: str | Path, indent: int = 2) -> None:
    """Save data as a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)


# ──── TSV/CSV I/O ────


def load_tsv(path: str | Path) -> pd.DataFrame:
    """Load a TSV file (BIDS-style metadata)."""
    return pd.read_csv(path, sep="\t")


def save_tsv(df: pd.DataFrame, path: str | Path) -> None:
    """Save a DataFrame as TSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file."""
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Save a DataFrame as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ──── NumPy I/O ────


def load_epochs(path: str | Path) -> np.ndarray:
    """Load preprocessed EEG epochs from .npy file.

    Expected shape: (N_trials, C_channels, T_samples)
    """
    data = np.load(path)
    if data.ndim != 3:
        raise ValueError(
            f"Expected 3D array (trials, channels, samples), got shape {data.shape}"
        )
    return data


def save_epochs(data: np.ndarray, path: str | Path) -> None:
    """Save preprocessed EEG epochs to .npy file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, data)


def load_labels(path: str | Path) -> np.ndarray:
    """Load label array from .npy file."""
    return np.load(path)


# ──── Split I/O ────


def load_split(path: str | Path) -> dict[str, list]:
    """Load a train/val/test split file (JSON).

    Expected format:
    {
        "train": ["trial_00001", "trial_00002", ...],
        "val": ["trial_00050", ...],
        "test": ["trial_00080", ...]
    }
    """
    return load_json(path)


def save_split(split: dict[str, list], path: str | Path) -> None:
    """Save a split file."""
    save_json(split, path)


# ──── Manifest I/O ────


def load_manifest(path: str | Path) -> dict:
    """Load the dataset manifest."""
    return load_json(path)


# ──── Path utilities ────


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist and return the Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_files(directory: str | Path, pattern: str = "*.npy") -> list[Path]:
    """Recursively find files matching a glob pattern."""
    return sorted(Path(directory).rglob(pattern))
