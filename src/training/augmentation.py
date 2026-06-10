"""
EEG data augmentation transforms for self-supervised and supervised training.
"""

from __future__ import annotations

import random
from typing import Literal

import numpy as np
import torch


class TimeShift:
    """Random circular time shift.

    Parameters
    ----------
    max_shift_samples : int
        Maximum shift in samples (both directions).
    """

    def __init__(self, max_shift_samples: int = 25):
        self.max_shift = max_shift_samples

    def __call__(self, x: np.ndarray) -> np.ndarray:
        shift = random.randint(-self.max_shift, self.max_shift)
        return np.roll(x, shift, axis=-1)


class ChannelDropout:
    """Randomly zero-out entire channels.

    Parameters
    ----------
    p : float
        Probability of dropping each channel.
    """

    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, x: np.ndarray) -> np.ndarray:
        mask = np.random.binomial(1, 1 - self.p, size=(x.shape[0], 1)).astype(x.dtype)
        return x * mask


class GaussianNoise:
    """Add Gaussian noise to EEG signal.

    Parameters
    ----------
    std : float
        Standard deviation of noise (relative to signal std).
    """

    def __init__(self, std: float = 0.1):
        self.std = std

    def __call__(self, x: np.ndarray) -> np.ndarray:
        noise = np.random.randn(*x.shape).astype(x.dtype) * self.std
        return x + noise


class FrequencyMasking:
    """Mask random frequency bands in the spectrogram.

    Applied AFTER time-frequency decomposition (not to raw EEG).

    Parameters
    ----------
    max_bands : int
        Maximum number of frequency bands to mask.
    max_width : int
        Maximum width of each masked band (in frequency bins).
    """

    def __init__(self, max_bands: int = 2, max_width: int = 10):
        self.max_bands = max_bands
        self.max_width = max_width

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """x: shape (..., F, T) — frequency × time."""
        n_freqs = x.shape[-2]
        for _ in range(random.randint(1, self.max_bands)):
            width = random.randint(1, self.max_width)
            start = random.randint(0, max(0, n_freqs - width))
            x[..., start : start + width, :] = 0
        return x


class TimeMasking:
    """Mask random time segments.

    Parameters
    ----------
    max_width : int
        Maximum width of masked segment (in time samples).
    max_masks : int
        Maximum number of masks to apply.
    """

    def __init__(self, max_width: int = 50, max_masks: int = 2):
        self.max_width = max_width
        self.max_masks = max_masks

    def __call__(self, x: np.ndarray) -> np.ndarray:
        n_time = x.shape[-1]
        for _ in range(random.randint(1, self.max_masks)):
            width = random.randint(1, self.max_width)
            start = random.randint(0, max(0, n_time - width))
            x[..., start : start + width] = 0
        return x


class AmplitudeScale:
    """Random amplitude scaling per-trial.

    Parameters
    ----------
    scale_range : tuple[float, float]
        Min and max scale factors.
    """

    def __init__(self, scale_range: tuple[float, float] = (0.8, 1.2)):
        self.low, self.high = scale_range

    def __call__(self, x: np.ndarray) -> np.ndarray:
        scale = random.uniform(self.low, self.high)
        return x * scale


class ChannelPermutation:
    """Randomly permute channel order.

    This is a strong augmentation that breaks spatial structure.
    Use with caution — only appropriate for models that should be
    spatially invariant.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        perm = np.random.permutation(x.shape[0])
        return x[perm]


class MixupTransform:
    """Mixup augmentation (applied at batch level, not per-sample).

    Parameters
    ----------
    alpha : float
        Beta distribution parameter.
    """

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply mixup to a batch.

        Parameters
        ----------
        x : torch.Tensor
            Batch input, shape (batch, C, T).
        y : torch.Tensor
            Batch labels, shape (batch,).

        Returns
        -------
        tuple of (mixed_x, y_a, y_b, lambda)
        """
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam


class Compose:
    """Compose multiple augmentation transforms.

    Parameters
    ----------
    transforms : list
        List of transform objects.
    p : float
        Probability of applying each transform.
    """

    def __init__(self, transforms: list, p: float = 1.0):
        self.transforms = transforms
        self.p = p

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            if random.random() < self.p:
                x = t(x)
        return x
