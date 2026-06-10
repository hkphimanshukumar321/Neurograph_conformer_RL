"""
Re-referencing — Common Average Reference (CAR) and other schemes.
"""

from __future__ import annotations

import logging

import mne

logger = logging.getLogger(__name__)


def apply_car(raw: mne.io.Raw) -> mne.io.Raw:
    """Apply Common Average Reference (CAR).

    CAR is montage-agnostic and does not require specific reference electrodes
    (e.g., mastoids). This makes it suitable for cross-dataset harmonization
    where reference schemes vary (e.g., Emotiv EPOC has no earlobe reference).

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data.

    Returns
    -------
    mne.io.Raw
        Re-referenced data.
    """
    logger.info("Applying Common Average Reference (CAR)")
    raw.set_eeg_reference("average", projection=False)
    return raw


def apply_custom_reference(
    raw: mne.io.Raw,
    ref_channels: list[str],
) -> mne.io.Raw:
    """Apply custom reference using specified channels.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data.
    ref_channels : list[str]
        Channel names to use as reference (e.g., ['M1', 'M2'] for mastoids).

    Returns
    -------
    mne.io.Raw
        Re-referenced data.
    """
    available = set(raw.ch_names)
    missing = [ch for ch in ref_channels if ch not in available]
    if missing:
        logger.warning(
            f"Reference channels {missing} not found. Falling back to CAR."
        )
        return apply_car(raw)

    logger.info(f"Applying custom reference: {ref_channels}")
    raw.set_eeg_reference(ref_channels=ref_channels)
    return raw
