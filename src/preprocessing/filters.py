import numpy as np
import mne
import logging

logger = logging.getLogger(__name__)

class EEGFilter:
    """
    A wrapper around MNE's filtering functions to apply consistent DSP logic
    to the EEG signals across different datasets.
    """
    def __init__(self, target_sfreq=250.0, l_freq=0.5, h_freq=100.0, notch_freqs=None):
        self.target_sfreq = target_sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq
        # Apply a dual notch by default to catch both US (60Hz) and EU/CN (50Hz) line noise,
        # plus their harmonics (100Hz, 120Hz). We limit to 50 and 60 to avoid excessive comb filtering.
        self.notch_freqs = notch_freqs if notch_freqs is not None else [50.0, 60.0]
        
    def __call__(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """
        Applies filtering and resampling on an mne.io.Raw object.
        Operates mostly in-place to save memory.
        """
        raw.load_data()
        
        # 1. Notch filter (remove power line noise)
        try:
            # Check if sfreq is high enough for the notch filters
            valid_freqs = [f for f in self.notch_freqs if f < raw.info['sfreq'] / 2.0]
            if valid_freqs:
                raw.notch_filter(freqs=valid_freqs, filter_length='auto', phase='zero', verbose=False)
        except Exception as e:
            logger.warning(f"Failed to apply notch filter: {e}")
            
        # 2. Bandpass filter (remove drift and high-freq noise)
        try:
            raw.filter(l_freq=self.l_freq, h_freq=self.h_freq, filter_length='auto', phase='zero', verbose=False)
        except Exception as e:
            logger.warning(f"Failed to apply bandpass filter: {e}")
            
        # 3. Resample to common sampling rate
        current_sfreq = raw.info['sfreq']
        if not np.isclose(current_sfreq, self.target_sfreq):
            raw.resample(sfreq=self.target_sfreq, verbose=False)
            
        return raw
