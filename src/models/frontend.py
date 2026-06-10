"""
Front-end modules — time-frequency decomposition of raw EEG.

Two variants:
  1. WaveletFrontEnd: Morlet CWT (offline, precomputed or on-the-fly)
  2. FilterBankFrontEnd: Learnable SincNet-style band-pass filters (end-to-end)
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveletFrontEnd(nn.Module):
    """Continuous Wavelet Transform (Morlet) front-end.

    Converts raw EEG (batch, C, T) to time-frequency representation
    (batch, C, F, T'). Uses precomputed wavelet kernels for efficiency.

    Parameters
    ----------
    n_freqs : int
        Number of frequency bins.
    freq_min : float
        Minimum frequency in Hz.
    freq_max : float
        Maximum frequency in Hz.
    sfreq : float
        Sampling frequency in Hz.
    n_cycles_mode : str
        'adaptive' (n_cycles = freq / 2) or 'fixed'.
    n_cycles_fixed : float
        Fixed number of cycles (used if mode='fixed').
    time_downsample : int
        Downsample factor for temporal axis.
    d_out : int
        Output projection dimension. If > 0, project (F, T') → d_out.
    """

    def __init__(
        self,
        n_freqs: int = 100,
        freq_min: float = 1.0,
        freq_max: float = 100.0,
        sfreq: float = 250.0,
        n_cycles_mode: str = "adaptive",
        n_cycles_fixed: float = 7.0,
        time_downsample: int = 4,
        d_out: int = 0,
    ):
        super().__init__()
        self.n_freqs = n_freqs
        self.sfreq = sfreq
        self.time_downsample = time_downsample

        # Compute frequencies
        freqs = np.linspace(freq_min, freq_max, n_freqs)
        self.register_buffer("freqs", torch.tensor(freqs, dtype=torch.float32))

        # Compute n_cycles per frequency
        if n_cycles_mode == "adaptive":
            n_cycles = freqs / 2.0
        else:
            n_cycles = np.full(n_freqs, n_cycles_fixed)

        # Precompute Morlet wavelet kernels
        # Each kernel is a complex 1D convolution filter
        kernels_real = []
        kernels_imag = []
        max_len = 0

        for i, (f, nc) in enumerate(zip(freqs, n_cycles)):
            sigma_t = nc / (2 * np.pi * f)
            t_range = int(6 * sigma_t * sfreq)  # ±3 sigma
            t_range = max(t_range, 3)  # minimum kernel size
            if t_range % 2 == 0:
                t_range += 1
            max_len = max(max_len, t_range)

        # Pad all kernels to same length for batched conv
        self.kernel_len = max_len if max_len % 2 == 1 else max_len + 1
        padding = self.kernel_len // 2

        for i, (f, nc) in enumerate(zip(freqs, n_cycles)):
            sigma_t = nc / (2 * np.pi * f)
            t = torch.arange(-(self.kernel_len // 2), self.kernel_len // 2 + 1, dtype=torch.float32) / sfreq
            gaussian = torch.exp(-t**2 / (2 * sigma_t**2))
            # Normalize
            gaussian = gaussian / gaussian.sum()
            # Complex sinusoid
            real_part = gaussian * torch.cos(2 * np.pi * f * t)
            imag_part = gaussian * torch.sin(2 * np.pi * f * t)
            kernels_real.append(real_part)
            kernels_imag.append(imag_part)

        # Stack: (F, 1, kernel_len) for grouped conv
        kernels_real = torch.stack(kernels_real).unsqueeze(1)  # (F, 1, K)
        kernels_imag = torch.stack(kernels_imag).unsqueeze(1)  # (F, 1, K)
        self.register_buffer("kernels_real", kernels_real)
        self.register_buffer("kernels_imag", kernels_imag)
        self.padding = padding

        # Optional output projection
        self.projection = None
        if d_out > 0:
            # After CWT: (batch, C, F, T') → flatten F,T' per channel → project
            self.projection = nn.LazyLinear(d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute CWT of input EEG.

        Parameters
        ----------
        x : torch.Tensor
            Raw EEG, shape (batch, C, T).

        Returns
        -------
        torch.Tensor
            Time-frequency representation:
            - If d_out=0: shape (batch, C, F, T')
            - If d_out>0: shape (batch, C, d_out)
        """
        batch, n_ch, n_time = x.shape

        # Reshape: treat each channel independently
        # (batch * C, 1, T) for conv1d
        x_flat = x.reshape(batch * n_ch, 1, n_time)

        # Expand to (batch*C, F, T) by repeating input for each freq
        x_expand = x_flat.expand(-1, self.n_freqs, -1)
        x_expand = x_expand.reshape(batch * n_ch * self.n_freqs, 1, n_time)

        # Expand kernels: (F, 1, K) → (batch*C*F, 1, K)
        kr = self.kernels_real.repeat(batch * n_ch, 1, 1)
        ki = self.kernels_imag.repeat(batch * n_ch, 1, 1)

        # Grouped conv1d
        real = F.conv1d(x_expand, kr, padding=self.padding, groups=batch * n_ch * self.n_freqs)
        imag = F.conv1d(x_expand, ki, padding=self.padding, groups=batch * n_ch * self.n_freqs)

        # Power: |z|^2 = real^2 + imag^2
        power = real.squeeze(1) ** 2 + imag.squeeze(1) ** 2  # (batch*C*F, T)
        power = power.reshape(batch, n_ch, self.n_freqs, n_time)

        # Temporal downsampling
        if self.time_downsample > 1:
            power = F.avg_pool2d(
                power.reshape(batch * n_ch, self.n_freqs, n_time).unsqueeze(1),
                kernel_size=(1, self.time_downsample),
                stride=(1, self.time_downsample),
            ).squeeze(1)
            t_out = power.shape[-1]
            power = power.reshape(batch, n_ch, self.n_freqs, t_out)

        # Log-scale for better dynamic range
        power = torch.log1p(power)

        if self.projection is not None:
            # Flatten freq and time dims, project per channel
            power_flat = power.reshape(batch * n_ch, -1)
            projected = self.projection(power_flat)
            return projected.reshape(batch, n_ch, -1)

        return power


class FilterBankFrontEnd(nn.Module):
    """Learnable filter-bank front-end (SincNet-inspired).

    Uses parameterized sinc functions as 1D convolution kernels. The network
    learns optimal band-pass cutoff frequencies end-to-end.

    Parameters
    ----------
    n_bands : int
        Number of frequency bands.
    kernel_size : int
        Size of sinc filter kernel.
    sfreq : float
        Sampling frequency in Hz.
    d_out : int
        Output feature dimension per band (via log-power + pooling).
    """

    def __init__(
        self,
        n_bands: int = 6,
        kernel_size: int = 251,
        sfreq: float = 250.0,
        d_out: int = 64,
        init_bands: list[tuple[float, float]] | None = None,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.sfreq = sfreq
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

        # Initialize band edges
        if init_bands is None:
            init_bands = [
                (0.5, 4), (4, 8), (8, 13), (13, 30), (30, 50), (50, 100)
            ]
            init_bands = init_bands[:n_bands]

        # Learnable parameters: low and high cutoff frequencies (in Hz)
        low_freqs = torch.tensor([b[0] for b in init_bands], dtype=torch.float32)
        band_widths = torch.tensor([b[1] - b[0] for b in init_bands], dtype=torch.float32)
        self.low_hz = nn.Parameter(low_freqs)
        self.band_hz = nn.Parameter(band_widths)

        # Hamming window
        n = torch.arange(0, self.kernel_size, dtype=torch.float32)
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * n / (self.kernel_size - 1))
        self.register_buffer("window", window)

        # Output projection
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(n_bands, d_out) if d_out > 0 else nn.Identity()

    def _compute_filters(self) -> torch.Tensor:
        """Compute sinc band-pass filters from learnable parameters.

        Returns
        -------
        torch.Tensor
            Filter kernels, shape (n_bands, 1, kernel_size).
        """
        nyquist = self.sfreq / 2.0
        low = torch.clamp(self.low_hz, 0.1, nyquist - 1) / nyquist
        high = torch.clamp(low + torch.abs(self.band_hz), low + 0.01, 1.0)

        # Sinc filter: bandpass = lowpass(high) - lowpass(low)
        n = torch.arange(0, self.kernel_size, dtype=torch.float32, device=low.device)
        n = n - (self.kernel_size - 1) / 2.0
        n = n.unsqueeze(0)  # (1, K)

        filters = []
        for i in range(self.n_bands):
            # Low-pass at high freq
            sinc_high = 2 * high[i] * torch.sinc(2 * high[i] * n)
            # Low-pass at low freq
            sinc_low = 2 * low[i] * torch.sinc(2 * low[i] * n)
            # Band-pass
            bp = (sinc_high - sinc_low) * self.window.unsqueeze(0)
            # Normalize
            bp = bp / bp.abs().sum(dim=-1, keepdim=True)
            filters.append(bp)

        return torch.cat(filters, dim=0).unsqueeze(1)  # (n_bands, 1, K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learnable filter bank.

        Parameters
        ----------
        x : torch.Tensor
            Raw EEG, shape (batch, C, T).

        Returns
        -------
        torch.Tensor
            Band-power features, shape (batch, C, n_bands) or (batch, C, d_out).
        """
        batch, n_ch, n_time = x.shape
        filters = self._compute_filters()  # (n_bands, 1, K)

        # Apply each filter to each channel
        x_flat = x.reshape(batch * n_ch, 1, n_time)  # (B*C, 1, T)
        padding = self.kernel_size // 2

        # Convolve: (B*C, n_bands, T)
        filtered = F.conv1d(x_flat, filters, padding=padding)

        # Log-power
        power = torch.log1p(filtered ** 2)

        # Pool over time: (B*C, n_bands)
        pooled = self.pool(power).squeeze(-1)

        # Reshape: (B, C, n_bands)
        pooled = pooled.reshape(batch, n_ch, self.n_bands)

        return self.proj(pooled)
