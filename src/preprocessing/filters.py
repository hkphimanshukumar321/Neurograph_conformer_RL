"""
Filtering operations — bandpass, notch, and resampling.
"""

from __future__ import annotations

import logging

import mne
import numpy as np

logger = logging.getLogger(__name__)


def bandpass_filter(
    raw: mne.io.Raw,
    l_freq: float = 0.5,
    h_freq: float = 100.0,
    method: str = "fir",
    phase: str = "zero-double",
) -> mne.io.Raw:
    """Apply bandpass filter to raw EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data.
    l_freq : float
        Low cutoff frequency in Hz.
    h_freq : float
        High cutoff frequency in Hz.
    method : str
        Filter method: 'fir' or 'iir'.
    phase : str
        Filter phase: 'zero', 'zero-double', 'minimum'.

    Returns
    -------
    mne.io.Raw
        Filtered data (modified in-place).
    """
    logger.info(f"Bandpass filter: {l_freq}–{h_freq} Hz (method={method})")
    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method=method,
        phase=phase,
        fir_design="firwin",
        picks="eeg",
    )
    return raw


def notch_filter(
    raw: mne.io.Raw,
    freqs: list[float] | float = 50.0,
    notch_widths: float = 2.0,
) -> mne.io.Raw:
    """Apply notch filter to remove powerline noise.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data.
    freqs : float or list[float]
        Notch frequencies in Hz (e.g., [50, 100] or [60, 120]).
    notch_widths : float
        Width of the notch in Hz.

    Returns
    -------
    mne.io.Raw
        Filtered data.
    """
    if isinstance(freqs, (int, float)):
        freqs = [freqs]
    logger.info(f"Notch filter at {freqs} Hz (width={notch_widths} Hz)")
    raw.notch_filter(freqs=freqs, notch_widths=notch_widths, picks="eeg")
    return raw


def resample(
    raw: mne.io.Raw,
    sfreq: float = 250.0,
) -> mne.io.Raw:
    """Resample raw EEG data to target sampling frequency.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data.
    sfreq : float
        Target sampling frequency in Hz.

    Returns
    -------
    mne.io.Raw
        Resampled data.
    """
    original_sfreq = raw.info["sfreq"]
    if abs(original_sfreq - sfreq) < 0.01:
        logger.info(f"Already at target sfreq={sfreq} Hz, skipping resample")
        return raw

    logger.info(f"Resampling: {original_sfreq} Hz → {sfreq} Hz")
    raw.resample(sfreq=sfreq, npad="auto")
    return raw


def causal_bandpass_filter(
    data: np.ndarray,
    sfreq: float,
    l_freq: float = 0.5,
    h_freq: float = 100.0,
    order: int = 5,
) -> np.ndarray:
    """Apply causal (forward-only) IIR bandpass filter for real-time use.

    Unlike zero-phase filtering, this introduces phase delay but is
    causal and suitable for streaming/real-time applications.

    Parameters
    ----------
    data : np.ndarray
        EEG data, shape (C, T) or (T,).
    sfreq : float
        Sampling frequency in Hz.
    l_freq : float
        Low cutoff frequency.
    h_freq : float
        High cutoff frequency.
    order : int
        Filter order.

    Returns
    -------
    np.ndarray
        Filtered data, same shape as input.
    """
    from scipy.signal import butter, lfilter

    nyq = sfreq / 2.0
    low = l_freq / nyq
    high = h_freq / nyq
    b, a = butter(order, [low, high], btype="band")
    return lfilter(b, a, data, axis=-1)
