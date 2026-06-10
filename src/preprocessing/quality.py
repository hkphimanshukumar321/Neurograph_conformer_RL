"""
Quality control — compute and report preprocessing quality metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Preprocessing quality metrics for a single subject/session."""

    n_epochs_total: int = 0
    n_epochs_kept: int = 0
    n_epochs_rejected: int = 0
    rejection_rate: float = 0.0
    n_bad_channels: int = 0
    bad_channels: list[str] = field(default_factory=list)
    n_ica_components_removed: int = 0
    mean_amplitude_uv: float = 0.0
    std_amplitude_uv: float = 0.0
    max_amplitude_uv: float = 0.0
    per_channel_snr: dict[str, float] = field(default_factory=dict)
    flag: str = "good"  # "good", "marginal", "poor"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "n_epochs_total": self.n_epochs_total,
            "n_epochs_kept": self.n_epochs_kept,
            "n_epochs_rejected": self.n_epochs_rejected,
            "rejection_rate": round(self.rejection_rate, 4),
            "n_bad_channels": self.n_bad_channels,
            "bad_channels": self.bad_channels,
            "n_ica_components_removed": self.n_ica_components_removed,
            "mean_amplitude_uv": round(self.mean_amplitude_uv, 2),
            "std_amplitude_uv": round(self.std_amplitude_uv, 2),
            "max_amplitude_uv": round(self.max_amplitude_uv, 2),
            "flag": self.flag,
            "warnings": self.warnings,
        }


def compute_quality_metrics(
    data: np.ndarray,
    labels: np.ndarray,
    sfreq: float,
    ch_names: list[str] | None = None,
) -> QualityReport:
    """Compute quality metrics on preprocessed epoch data.

    Parameters
    ----------
    data : np.ndarray
        Preprocessed epochs, shape (N_trials, C, T).
    labels : np.ndarray
        Labels, shape (N_trials,).
    sfreq : float
        Sampling frequency.
    ch_names : list[str], optional
        Channel names.

    Returns
    -------
    QualityReport
        Quality metrics.
    """
    n_trials, n_channels, n_samples = data.shape

    # Amplitude statistics (convert back to µV scale if normalized)
    mean_amp = np.mean(np.abs(data))
    std_amp = np.std(data)
    max_amp = np.max(np.abs(data))

    # Per-channel SNR estimate (signal variance / noise variance)
    # Rough estimate: signal = trial-average ERP, noise = residual
    per_channel_snr = {}
    if ch_names and n_trials > 1:
        for ch_idx in range(min(n_channels, len(ch_names or []))):
            ch_data = data[:, ch_idx, :]  # (N_trials, T)
            signal = np.mean(ch_data, axis=0)  # ERP
            noise = ch_data - signal[np.newaxis, :]
            signal_power = np.var(signal)
            noise_power = np.mean(np.var(noise, axis=1))
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            if ch_names:
                per_channel_snr[ch_names[ch_idx]] = round(float(snr), 2)

    # Quality flag
    report = QualityReport(
        n_epochs_kept=n_trials,
        mean_amplitude_uv=float(mean_amp),
        std_amplitude_uv=float(std_amp),
        max_amplitude_uv=float(max_amp),
        per_channel_snr=per_channel_snr,
    )

    # Check for issues
    if n_trials < 10:
        report.warnings.append(f"Very few epochs: {n_trials}")
        report.flag = "poor"
    if max_amp > 500:
        report.warnings.append(f"High max amplitude: {max_amp:.1f} µV — possible artifact")
        report.flag = "marginal"

    # Class balance check
    unique, counts = np.unique(labels, return_counts=True)
    min_count = counts.min()
    max_count = counts.max()
    if max_count > 3 * min_count:
        report.warnings.append(
            f"Severe class imbalance: min={min_count}, max={max_count}"
        )

    return report
