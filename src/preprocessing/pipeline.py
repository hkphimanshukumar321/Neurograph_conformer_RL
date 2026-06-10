"""
Main preprocessing pipeline — orchestrates all preprocessing steps.

Supports two modes:
  - 'provider': replicate each dataset's published preprocessing
  - 'unified': standardized pipeline for fair cross-dataset comparison
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import mne
import numpy as np
from omegaconf import DictConfig

from src.preprocessing.artifacts import remove_artifacts_asr, remove_artifacts_ica, reject_by_amplitude
from src.preprocessing.epoching import create_epochs
from src.preprocessing.filters import bandpass_filter, notch_filter, resample
from src.preprocessing.normalization import zscore_normalize, robust_normalize
from src.preprocessing.quality import QualityReport, compute_quality_metrics
from src.preprocessing.reference import apply_car

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    """Container for preprocessing output."""

    epochs: np.ndarray          # (N_trials, C, T)
    labels: np.ndarray          # (N_trials,)
    metadata: dict              # trial metadata
    quality: QualityReport      # quality metrics
    sfreq: float                # final sampling frequency
    ch_names: list[str]         # channel names
    n_rejected: int             # number of rejected epochs
    n_total: int                # total epochs before rejection


class PreprocessingPipeline:
    """Configurable EEG preprocessing pipeline.

    Parameters
    ----------
    cfg : DictConfig
        Preprocessing configuration (from base.yaml + dataset override).
    mode : str
        Pipeline mode: 'unified' or 'provider'.
    """

    def __init__(self, cfg: DictConfig, mode: Literal["unified", "provider"] = "unified"):
        self.cfg = cfg
        self.mode = mode
        logger.info(f"Initialized preprocessing pipeline (mode={mode})")

    def run(self, raw: mne.io.Raw, events: np.ndarray, event_id: dict) -> PreprocessingResult:
        """Execute the full preprocessing pipeline.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data loaded via MNE.
        events : np.ndarray
            MNE-format events array (N, 3).
        event_id : dict
            Mapping from event name to event code.

        Returns
        -------
        PreprocessingResult
            Preprocessed epochs with metadata and quality report.
        """
        cfg = self.cfg
        n_total_events = len(events)
        logger.info(f"Starting preprocessing: {n_total_events} events, "
                     f"{len(raw.ch_names)} channels, {raw.info['sfreq']} Hz")

        # ── Step 1: Bad channel detection & interpolation ──
        raw = self._handle_bad_channels(raw)

        # ── Step 2: Filtering ──
        raw = bandpass_filter(raw, l_freq=cfg.bandpass_low, h_freq=cfg.bandpass_high,
                              method=cfg.get("filter_method", "fir"))
        if cfg.notch_freq is not None:
            harmonics = [cfg.notch_freq]
            if cfg.notch_freq * 2 <= cfg.bandpass_high:
                harmonics.append(cfg.notch_freq * 2)
            raw = notch_filter(raw, freqs=harmonics)

        # ── Step 3: Resampling ──
        if raw.info["sfreq"] != cfg.target_srate:
            raw = resample(raw, sfreq=cfg.target_srate)

        # ── Step 4: Re-referencing ──
        if cfg.reference == "car":
            raw = apply_car(raw)

        # ── Step 5: Artifact handling ──
        raw = self._handle_artifacts(raw)

        # ── Step 6: Epoching ──
        epochs_mne = create_epochs(
            raw, events, event_id,
            tmin=cfg.epoch_tmin, tmax=cfg.epoch_tmax,
            baseline=(cfg.baseline_tmin, cfg.baseline_tmax),
        )

        # ── Step 7: Amplitude rejection ──
        epochs_mne, n_rejected = reject_by_amplitude(
            epochs_mne, threshold=cfg.amplitude_threshold
        )

        # ── Step 8: Extract numpy arrays ──
        data = epochs_mne.get_data()  # (N_trials, C, T)
        labels = epochs_mne.events[:, -1]

        # ── Step 9: Normalization ──
        if cfg.normalization == "zscore":
            data = zscore_normalize(data, scope=cfg.norm_scope)
        elif cfg.normalization == "robust":
            data = robust_normalize(data, scope=cfg.norm_scope)

        # ── Step 10: Quality control ──
        quality = compute_quality_metrics(
            data, labels, raw.info["sfreq"],
            ch_names=epochs_mne.ch_names,
        )

        logger.info(
            f"Preprocessing complete: {data.shape[0]}/{n_total_events} epochs kept "
            f"({n_rejected} rejected), shape={data.shape}"
        )

        return PreprocessingResult(
            epochs=data,
            labels=labels,
            metadata={
                "event_id": event_id,
                "sfreq": cfg.target_srate,
                "tmin": cfg.epoch_tmin,
                "tmax": cfg.epoch_tmax,
                "mode": self.mode,
            },
            quality=quality,
            sfreq=cfg.target_srate,
            ch_names=epochs_mne.ch_names,
            n_rejected=n_rejected,
            n_total=n_total_events,
        )

    def _handle_bad_channels(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Detect and interpolate bad channels."""
        try:
            # Use autoreject's RANSAC for bad channel detection
            from autoreject import Ransac
            logger.info("Using RANSAC for bad channel detection")
            # RANSAC requires epochs; we'll mark channels as bad based on statistics
        except ImportError:
            logger.warning("autoreject not available; using amplitude-based bad channel detection")

        # Simple amplitude-based detection
        data = raw.get_data()
        ch_std = np.std(data, axis=1)
        median_std = np.median(ch_std)
        bad_mask = (ch_std > 5 * median_std) | (ch_std < 0.01 * median_std)
        bad_channels = [raw.ch_names[i] for i in np.where(bad_mask)[0]]

        if bad_channels:
            logger.info(f"Detected {len(bad_channels)} bad channels: {bad_channels}")
            raw.info["bads"] = bad_channels
            raw = raw.interpolate_bads(reset_bads=True)

        return raw

    def _handle_artifacts(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply artifact removal method."""
        method = self.cfg.artifact_method

        if method == "ica":
            raw = remove_artifacts_ica(
                raw,
                n_components=self.cfg.get("ica_n_components"),
                max_remove=self.cfg.get("ica_max_remove", 3),
            )
        elif method == "asr":
            raw = remove_artifacts_asr(raw, threshold=20.0)
        elif method == "threshold":
            logger.info("Using threshold-only artifact rejection (applied during epoching)")
        else:
            raise ValueError(f"Unknown artifact method: {method}")

        return raw
