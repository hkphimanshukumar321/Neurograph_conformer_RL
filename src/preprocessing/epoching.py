"""
Epoching — segment continuous EEG into event-locked epochs.
"""

from __future__ import annotations

import logging
from typing import Optional

import mne
import numpy as np

logger = logging.getLogger(__name__)


def create_epochs(
    raw: mne.io.Raw,
    events: np.ndarray,
    event_id: dict,
    tmin: float = 0.0,
    tmax: float = 2.0,
    baseline: Optional[tuple[float, float]] = (-0.5, 0.0),
    preload: bool = True,
) -> mne.Epochs:
    """Create MNE Epochs from raw data and events.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data.
    events : np.ndarray
        Events array, shape (N, 3).
    event_id : dict
        Mapping from event name to event code.
    tmin : float
        Start time of epoch relative to event, in seconds.
    tmax : float
        End time of epoch relative to event, in seconds.
    baseline : tuple[float, float] or None
        Baseline correction interval. None to skip baseline correction.
    preload : bool
        Whether to preload data into memory.

    Returns
    -------
    mne.Epochs
        Epochs object.
    """
    logger.info(
        f"Creating epochs: tmin={tmin}, tmax={tmax}, baseline={baseline}, "
        f"n_events={len(events)}"
    )

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=preload,
        picks="eeg",
        proj=False,
        on_missing="warn",
    )

    logger.info(
        f"Created {len(epochs)} epochs, "
        f"shape=({len(epochs)}, {len(epochs.ch_names)}, {len(epochs.times)})"
    )

    return epochs


def create_sliding_window_epochs(
    data: np.ndarray,
    sfreq: float,
    window_s: float = 2.0,
    stride_s: float = 1.0,
) -> np.ndarray:
    """Create overlapping fixed-length windows from variable-length data.

    Used for sentence-level datasets (Chisco) where trial duration varies.

    Parameters
    ----------
    data : np.ndarray
        Single-trial EEG data, shape (C, T).
    sfreq : float
        Sampling frequency in Hz.
    window_s : float
        Window duration in seconds.
    stride_s : float
        Stride between windows in seconds.

    Returns
    -------
    np.ndarray
        Windowed data, shape (N_windows, C, window_samples).
    """
    window_samples = int(window_s * sfreq)
    stride_samples = int(stride_s * sfreq)
    n_channels, n_samples = data.shape

    if n_samples < window_samples:
        # Pad with zeros if trial is shorter than window
        padded = np.zeros((n_channels, window_samples), dtype=data.dtype)
        padded[:, :n_samples] = data
        return padded[np.newaxis, :, :]  # (1, C, window_samples)

    # Compute number of windows
    n_windows = (n_samples - window_samples) // stride_samples + 1

    windows = np.zeros((n_windows, n_channels, window_samples), dtype=data.dtype)
    for i in range(n_windows):
        start = i * stride_samples
        end = start + window_samples
        windows[i] = data[:, start:end]

    return windows
