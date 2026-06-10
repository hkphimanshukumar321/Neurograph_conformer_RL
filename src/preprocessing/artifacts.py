"""
Artifact removal — ICA, ASR, and amplitude-based rejection.
"""

from __future__ import annotations

import logging
from typing import Optional

import mne
import numpy as np

logger = logging.getLogger(__name__)


def remove_artifacts_ica(
    raw: mne.io.Raw,
    n_components: Optional[float | int] = None,
    max_remove: int = 3,
    method: str = "fastica",
    random_state: int = 42,
) -> mne.io.Raw:
    """Remove EOG/EMG artifacts using ICA.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data (should be filtered before ICA).
    n_components : float or int, optional
        Number of ICA components. If float (e.g., 0.999), uses variance explained.
        If None, defaults to n_channels - 1.
    max_remove : int
        Maximum number of components to remove.
    method : str
        ICA algorithm: 'fastica', 'infomax', 'picard'.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    mne.io.Raw
        Data with artifact components removed.
    """
    if n_components is None:
        n_components = 0.999  # variance-explained criterion

    logger.info(f"Running ICA (method={method}, n_components={n_components})")

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=method,
        random_state=random_state,
        max_iter="auto",
    )

    # Fit ICA on high-pass filtered copy (1 Hz) for better decomposition
    raw_for_ica = raw.copy().filter(l_freq=1.0, h_freq=None)
    ica.fit(raw_for_ica, picks="eeg")

    # Auto-detect EOG components
    eog_indices = []
    eog_channels = [ch for ch in raw.ch_names if ch.lower().startswith(("fp", "eog"))]
    if eog_channels:
        try:
            eog_indices, eog_scores = ica.find_bads_eog(
                raw, ch_name=eog_channels[0], threshold=3.0
            )
        except Exception as e:
            logger.warning(f"EOG detection failed: {e}")

    # Auto-detect EMG components (high-frequency power criterion)
    muscle_indices = []
    try:
        muscle_indices, muscle_scores = ica.find_bads_muscle(raw, threshold=0.5)
    except Exception as e:
        logger.warning(f"EMG detection failed: {e}")

    # Combine and limit
    bad_components = list(set(eog_indices + muscle_indices))
    if len(bad_components) > max_remove:
        logger.warning(
            f"Found {len(bad_components)} bad components, limiting to {max_remove}"
        )
        bad_components = bad_components[:max_remove]

    if bad_components:
        logger.info(f"Removing ICA components: {bad_components}")
        ica.exclude = bad_components
        raw = ica.apply(raw)
    else:
        logger.info("No artifact components detected by ICA")

    return raw


def remove_artifacts_asr(
    raw: mne.io.Raw,
    threshold: float = 20.0,
    window_s: float = 0.5,
) -> mne.io.Raw:
    """Remove artifacts using Artifact Subspace Reconstruction (ASR).

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data.
    threshold : float
        ASR threshold in standard deviations (default 20, conservative).
    window_s : float
        Window length in seconds.

    Returns
    -------
    mne.io.Raw
        Cleaned data.
    """
    try:
        from meegkit.asr import ASR

        logger.info(f"Running ASR (threshold={threshold} SD, window={window_s} s)")
        sfreq = raw.info["sfreq"]
        data = raw.get_data(picks="eeg")

        asr = ASR(sfreq=sfreq, cutoff=threshold)
        # Train on clean calibration data (first 60 seconds or all if shorter)
        calib_samples = min(int(60 * sfreq), data.shape[1])
        asr.fit(data[:, :calib_samples])

        # Transform
        clean_data = asr.transform(data)
        raw._data[: data.shape[0]] = clean_data

        logger.info("ASR artifact removal complete")
    except ImportError:
        logger.error("meegkit not installed — cannot use ASR. Install: pip install meegkit")
        raise

    return raw


def reject_by_amplitude(
    epochs: mne.Epochs,
    threshold: float = 100.0,
) -> tuple[mne.Epochs, int]:
    """Reject epochs exceeding amplitude threshold.

    Parameters
    ----------
    epochs : mne.Epochs
        MNE Epochs object.
    threshold : float
        Maximum peak-to-peak amplitude in µV.

    Returns
    -------
    tuple[mne.Epochs, int]
        (Cleaned epochs, number of rejected epochs).
    """
    n_before = len(epochs)
    reject_criteria = {"eeg": threshold * 1e-6}  # MNE expects V, not µV
    epochs.drop_bad(reject=reject_criteria)
    n_after = len(epochs)
    n_rejected = n_before - n_after

    logger.info(
        f"Amplitude rejection (±{threshold} µV): "
        f"{n_rejected}/{n_before} epochs rejected, {n_after} remaining"
    )
    return epochs, n_rejected
