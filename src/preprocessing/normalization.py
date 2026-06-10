"""
Normalization — per-channel, per-session z-score and robust normalization.

IMPORTANT: Never normalize across subjects or datasets. This would destroy
inter-subject variability that we want to model (or explicitly handle via
domain adaptation).
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


def zscore_normalize(
    data: np.ndarray,
    scope: Literal["per_session", "per_trial"] = "per_session",
    eps: float = 1e-8,
) -> np.ndarray:
    """Per-channel z-score normalization.

    Parameters
    ----------
    data : np.ndarray
        EEG epochs, shape (N_trials, C, T).
    scope : str
        - 'per_session': compute mean/std across all trials per channel
          (recommended for preserving cross-trial amplitude information).
        - 'per_trial': compute mean/std within each trial per channel
          (removes trial-level amplitude variation).
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    np.ndarray
        Normalized data, same shape as input.
    """
    data = data.astype(np.float32)

    if scope == "per_session":
        # Compute stats across all trials for each channel
        # data: (N, C, T) → mean/std over axes (0, 2) → shape (C,)
        mean = data.mean(axis=(0, 2), keepdims=True)  # (1, C, 1)
        std = data.std(axis=(0, 2), keepdims=True) + eps  # (1, C, 1)
        data = (data - mean) / std
        logger.debug(f"Z-score normalization (per_session): mean shape={mean.shape}")

    elif scope == "per_trial":
        # Compute stats within each trial for each channel
        # data: (N, C, T) → mean/std over axis 2 → shape (N, C)
        mean = data.mean(axis=2, keepdims=True)  # (N, C, 1)
        std = data.std(axis=2, keepdims=True) + eps  # (N, C, 1)
        data = (data - mean) / std
        logger.debug(f"Z-score normalization (per_trial): mean shape={mean.shape}")

    else:
        raise ValueError(f"Unknown normalization scope: {scope}")

    return data


def robust_normalize(
    data: np.ndarray,
    scope: Literal["per_session", "per_trial"] = "per_session",
    eps: float = 1e-8,
) -> np.ndarray:
    """Robust normalization using median and interquartile range (IQR).

    More robust to outliers than z-score normalization.

    Parameters
    ----------
    data : np.ndarray
        EEG epochs, shape (N_trials, C, T).
    scope : str
        Normalization scope (see zscore_normalize).
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    np.ndarray
        Normalized data.
    """
    data = data.astype(np.float32)

    if scope == "per_session":
        # Flatten trials and time: (N*T,) per channel
        n, c, t = data.shape
        flat = data.transpose(1, 0, 2).reshape(c, -1)  # (C, N*T)
        median = np.median(flat, axis=1).reshape(1, c, 1)
        q75 = np.percentile(flat, 75, axis=1).reshape(1, c, 1)
        q25 = np.percentile(flat, 25, axis=1).reshape(1, c, 1)
        iqr = (q75 - q25) + eps
        data = (data - median) / iqr

    elif scope == "per_trial":
        median = np.median(data, axis=2, keepdims=True)
        q75 = np.percentile(data, 75, axis=2, keepdims=True)
        q25 = np.percentile(data, 25, axis=2, keepdims=True)
        iqr = (q75 - q25) + eps
        data = (data - median) / iqr

    else:
        raise ValueError(f"Unknown normalization scope: {scope}")

    return data
