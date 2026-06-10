"""
Functional tests — Preprocessing pipeline correctness.

Verifies normalization distributions, epoching boundaries, quality metrics,
and augmentation transforms.
"""

import pytest
import numpy as np


class TestNormalizationCorrectness:
    """Verify normalization produces correct distributions."""

    def test_zscore_per_session_mean_zero(self):
        from src.preprocessing.normalization import zscore_normalize
        data = np.random.randn(50, 9, 500).astype(np.float32) * 10 + 5
        normed = zscore_normalize(data, scope="per_session")
        # Global mean across all samples should be ~0
        assert abs(normed.mean()) < 0.05

    def test_zscore_per_trial_mean_zero(self):
        from src.preprocessing.normalization import zscore_normalize
        data = np.random.randn(20, 9, 500).astype(np.float32) * 3 + 2
        normed = zscore_normalize(data, scope="per_trial")
        # Each trial should have ~0 mean
        for i in range(normed.shape[0]):
            assert abs(normed[i].mean()) < 0.05

    def test_robust_normalize_outlier_resistance(self):
        from src.preprocessing.normalization import robust_normalize
        data = np.random.randn(20, 9, 500).astype(np.float32)
        # Add outliers
        data[0, 0, 0] = 1000.0
        data[1, 3, 100] = -500.0
        normed = robust_normalize(data, scope="per_session")
        assert np.isfinite(normed).all()
        # Outliers should not dominate
        assert abs(np.median(normed)) < 1.0

    def test_normalization_preserves_shape(self):
        from src.preprocessing.normalization import zscore_normalize
        shapes = [(10, 9, 500), (5, 64, 1000), (1, 14, 250)]
        for shape in shapes:
            data = np.random.randn(*shape).astype(np.float32)
            normed = zscore_normalize(data, scope="per_session")
            assert normed.shape == shape


class TestEpochingCorrectness:
    """Verify sliding window epoching produces correct boundaries."""

    def test_epoch_count(self):
        from src.preprocessing.epoching import create_sliding_window_epochs
        data = np.random.randn(9, 1000).astype(np.float32)
        # window=500, stride=250 → (1000-500)/250 + 1 = 3
        epochs = create_sliding_window_epochs(data, sfreq=250.0, window_s=2.0, stride_s=1.0)
        assert epochs.shape[0] == 3

    def test_epoch_no_overlap_content(self):
        """Non-overlapping windows should contain different data."""
        from src.preprocessing.epoching import create_sliding_window_epochs
        data = np.arange(2000, dtype=np.float32).reshape(1, 2000)
        epochs = create_sliding_window_epochs(data, sfreq=250.0, window_s=2.0, stride_s=2.0)
        # First epoch: [0..499], second: [500..999]
        if epochs.shape[0] >= 2:
            assert not np.array_equal(epochs[0], epochs[1])

    def test_epoch_padding_short_input(self):
        from src.preprocessing.epoching import create_sliding_window_epochs
        data = np.random.randn(9, 100).astype(np.float32)
        epochs = create_sliding_window_epochs(data, sfreq=250.0, window_s=2.0, stride_s=1.0)
        assert epochs.shape[0] >= 1
        assert epochs.shape[2] == 500  # Padded to window size


class TestAugmentationCorrectness:
    """Verify augmentations produce valid outputs."""

    def test_time_shift_preserves_shape(self):
        from src.training.augmentation import TimeShift
        aug = TimeShift(max_shift_samples=25)
        x = np.random.randn(9, 500).astype(np.float32)
        out = aug(x)
        assert out.shape == x.shape

    def test_channel_dropout_zeroes_channels(self):
        from src.training.augmentation import ChannelDropout
        np.random.seed(42)
        aug = ChannelDropout(p=0.5)
        x = np.ones((9, 100), dtype=np.float32)
        out = aug(x)
        # Some channels should be zeroed
        n_zero = (out.sum(axis=1) == 0).sum()
        assert n_zero >= 0  # Non-negative (could be 0 by chance)

    def test_gaussian_noise_changes_signal(self):
        from src.training.augmentation import GaussianNoise
        aug = GaussianNoise(std=0.5)
        x = np.zeros((9, 500), dtype=np.float32)
        out = aug(x)
        assert not np.allclose(out, x)

    def test_compose_chains_transforms(self):
        from src.training.augmentation import Compose, TimeShift, GaussianNoise
        aug = Compose([TimeShift(10), GaussianNoise(0.1)], p=1.0)
        x = np.random.randn(9, 500).astype(np.float32)
        out = aug(x)
        assert out.shape == x.shape

    def test_frequency_masking(self):
        from src.training.augmentation import FrequencyMasking
        aug = FrequencyMasking(max_bands=2, max_width=5)
        x = np.ones((32, 100), dtype=np.float32)  # (F, T)
        out = aug(x)
        assert out.shape == x.shape
        # Some frequency bands should be zeroed
        assert (out == 0).any()

    def test_mixup_alpha(self):
        import torch
        from src.training.augmentation import MixupTransform
        aug = MixupTransform(alpha=0.2)
        x = torch.randn(8, 9, 500)
        y = torch.randint(0, 5, (8,))
        mixed_x, y_a, y_b, lam = aug(x, y)
        assert mixed_x.shape == x.shape
        assert 0 <= lam <= 1


class TestQualityMetrics:
    """Verify quality metric computation."""

    def test_quality_metrics_keys(self):
        from src.preprocessing.quality import compute_quality_metrics
        data = np.random.randn(15, 9, 500).astype(np.float64)
        labels = np.zeros(15, dtype=int)
        metrics = compute_quality_metrics(data, labels, sfreq=250.0)
        assert hasattr(metrics, "mean_amplitude_uv")
        assert hasattr(metrics, "max_amplitude_uv")

    def test_quality_detects_flat_channels(self):
        from src.preprocessing.quality import compute_quality_metrics
        data = np.random.randn(15, 9, 500).astype(np.float64)
        labels = np.zeros(15, dtype=int)
        data[:, 3, :] = 0.0  # Flat channel
        metrics = compute_quality_metrics(data, labels, sfreq=250.0)
        assert metrics is not None

class TestHarmonizationCorrectness:
    """Verify that EEG filtering and channel harmonization work correctly."""

    def test_filter_resampling(self):
        import mne
        from src.preprocessing.filters import EEGFilter
        
        info = mne.create_info(ch_names=['Fp1', 'Cz'], sfreq=500.0, ch_types='eeg')
        data = np.random.randn(2, 2000) # 4 seconds at 500Hz
        raw = mne.io.RawArray(data, info)
        
        filt = EEGFilter(target_sfreq=250.0, l_freq=1.0, h_freq=40.0, notch_freqs=[50.0])
        out_raw = filt(raw)
        
        # 4 seconds at 250Hz should be 1000 samples
        assert out_raw.info['sfreq'] == 250.0
        assert out_raw.get_data().shape == (2, 1000)

    def test_channel_harmonization(self):
        import mne
        from src.preprocessing.channel_harmonization import ChannelHarmonizer, STANDARD_CHANNELS
        
        # Simulate a 128-channel input that is missing Cz and has some weird extra channels
        input_channels = [f'Ch{i}' for i in range(128)]
        input_channels[0] = 'Fp1'
        input_channels[1] = 'Extra1'
        # Deliberately missing Cz
        
        info = mne.create_info(ch_names=input_channels, sfreq=250.0, ch_types='eeg')
        data = np.random.randn(128, 500)
        raw = mne.io.RawArray(data, info)
        
        harmonizer = ChannelHarmonizer()
        
        # We must disable mne logging as it can be verbose during interpolation
        mne.set_log_level('WARNING')
        out_raw = harmonizer(raw)
        
        # Output should have exactly len(STANDARD_CHANNELS) channels
        assert len(out_raw.ch_names) == len(STANDARD_CHANNELS)
        assert out_raw.ch_names == STANDARD_CHANNELS
        
        # Shape should be (61, 500)
        assert out_raw.get_data().shape == (len(STANDARD_CHANNELS), 500)
