"""
Electrode position utilities — standard 10-20 MNI coordinates and adjacency.

Provides functions to:
  1. Get 3D MNI coordinates for the standard 61-channel 10-20 layout
  2. Compute Gaussian-kernel distance-based adjacency matrices
  3. Compute correlation-based adjacency from batch EEG data

These are used by train.py to supply the adjacency matrix to the
GraphSpatialEncoder, replacing the dense all-ones fallback.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ── Standard 10-20 MNI coordinates (61 channels) ──────────────────────────
# These approximate MNI positions correspond to the STANDARD_CHANNELS list
# in src/preprocessing/channel_harmonization.py.
# Source: MNI coordinates from the standard_1020 montage in MNE-Python.

STANDARD_CHANNELS = [
    'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
    'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz',
    'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4',
    'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz',
    'PO4', 'PO8', 'O1', 'Oz', 'O2'
]


def get_standard_positions() -> np.ndarray:
    """Get approximate 3D positions for the 61 standard 10-20 electrodes.

    Uses MNE's standard_1020 montage to extract positions. Falls back to
    a spherical approximation if MNE is not available or fails.

    Returns
    -------
    np.ndarray
        Shape (61, 3) — XYZ coordinates in MNI space.
    """
    try:
        import mne
        montage = mne.channels.make_standard_montage('standard_1020')
        # Get positions for our 61 channels
        positions = []
        for ch in STANDARD_CHANNELS:
            ch_idx = montage.ch_names.index(ch)
            positions.append(montage.dig[ch_idx + 3].get('r', montage.get_positions()['ch_pos'][ch]))
        return np.array(positions, dtype=np.float32)
    except Exception:
        pass

    # Fallback: extract from MNE get_positions API
    try:
        import mne
        montage = mne.channels.make_standard_montage('standard_1020')
        ch_pos = montage.get_positions()['ch_pos']
        positions = []
        for ch in STANDARD_CHANNELS:
            if ch in ch_pos:
                positions.append(ch_pos[ch])
            else:
                # Use a default position if channel not found
                logger.warning(f"Channel {ch} not found in montage, using origin")
                positions.append(np.array([0.0, 0.0, 0.0]))
        return np.array(positions, dtype=np.float32)
    except Exception as e:
        logger.warning(f"MNE montage extraction failed: {e}. Using spherical approximation.")

    # Spherical fallback: place electrodes on a unit sphere
    return _spherical_positions(len(STANDARD_CHANNELS))


def _spherical_positions(n: int) -> np.ndarray:
    """Generate approximately uniform positions on a unit sphere.

    Uses the Fibonacci lattice for even distribution.

    Parameters
    ----------
    n : int
        Number of points.

    Returns
    -------
    np.ndarray
        Shape (n, 3).
    """
    golden_ratio = (1 + math.sqrt(5)) / 2
    positions = []
    for i in range(n):
        theta = math.acos(1 - 2 * (i + 0.5) / n)
        phi = 2 * math.pi * i / golden_ratio
        x = math.sin(theta) * math.cos(phi)
        y = math.sin(theta) * math.sin(phi)
        z = math.cos(theta)
        positions.append([x, y, z])
    return np.array(positions, dtype=np.float32)


def compute_distance_adjacency(
    positions: np.ndarray,
    sigma: float | None = None,
) -> torch.Tensor:
    """Compute Gaussian-kernel adjacency from 3D electrode positions.

    Parameters
    ----------
    positions : np.ndarray
        Shape (C, 3) — electrode positions.
    sigma : float, optional
        Gaussian kernel width. If None, uses the median inter-electrode distance.

    Returns
    -------
    torch.Tensor
        Shape (C, C) — adjacency matrix with values in [0, 1].
    """
    pos = torch.from_numpy(positions).float()
    # Pairwise Euclidean distance
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # (C, C, 3)
    dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)  # (C, C)

    # Auto sigma: median inter-electrode distance
    if sigma is None or sigma <= 0:
        nonzero = dist[dist > 1e-6]
        sigma = float(torch.median(nonzero).item()) if len(nonzero) > 0 else 1.0

    # Gaussian kernel
    adj = torch.exp(-dist ** 2 / (2 * sigma ** 2))

    logger.info(
        f"Computed distance adjacency: {adj.shape}, "
        f"sigma={sigma:.4f}, sparsity={float((adj < 0.1).sum()) / adj.numel():.1%}"
    )
    return adj


def compute_adjacency_for_batch(
    n_channels: int,
    device: torch.device,
    sigma: float | None = None,
) -> torch.Tensor:
    """Compute adjacency matrix for a given number of channels.

    If n_channels matches the standard 61-channel layout, uses MNI positions.
    Otherwise, uses spherical approximation.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    device : torch.device
        Target device.
    sigma : float, optional
        Gaussian kernel width.

    Returns
    -------
    torch.Tensor
        Shape (n_channels, n_channels) on `device`.
    """
    if n_channels == len(STANDARD_CHANNELS):
        positions = get_standard_positions()
    else:
        logger.info(
            f"Channel count {n_channels} != standard {len(STANDARD_CHANNELS)}. "
            f"Using spherical approximation for adjacency."
        )
        positions = _spherical_positions(n_channels)

    adj = compute_distance_adjacency(positions, sigma=sigma)
    return adj.to(device)
