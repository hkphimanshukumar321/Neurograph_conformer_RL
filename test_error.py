import mne
import numpy as np
from src.preprocessing.channel_harmonization import ChannelHarmonizer

# Create a dummy raw object with some weird scale or missing channels
ch_names = ['Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz'] # Only 6 channels
info = mne.create_info(ch_names=ch_names, sfreq=100, ch_types='eeg')
raw = mne.io.RawArray(np.zeros((len(ch_names), 1000)), info)

# Simulate wrong scale montage or no montage
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)

# Make coordinate weird
for ch in raw.info['chs']:
    ch['loc'][:3] *= 1000 # Make it in mm to simulate the 600cm radius error

harmonizer = ChannelHarmonizer()
harmonized_raw = harmonizer(raw)
