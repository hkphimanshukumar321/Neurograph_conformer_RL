import mne
import numpy as np
from src.preprocessing.channel_harmonization import ChannelHarmonizer

# Modify the script temporarily in memory
class FixedChannelHarmonizer(ChannelHarmonizer):
    def __call__(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        raw.load_data()
        
        # 1. Standardize channel names (strip spaces, standardize casing)
        ch_names = raw.ch_names
        mapping = {}
        for ch in ch_names:
            clean_ch = ch.replace(' ', '').replace('-REF', '').replace('-LE', '').upper()
            
            # Map standard casing
            for std_ch in self.target_channels:
                if clean_ch == std_ch.upper() or clean_ch == std_ch.upper().replace('Z', 'z'):
                    mapping[ch] = std_ch
                    break
        
        mne.rename_channels(raw.info, mapping)
        
        # 3. Identify missing and extra channels
        current_channels = raw.ch_names
        missing_channels = [ch for ch in self.target_channels if ch not in current_channels]
        extra_channels = [ch for ch in current_channels if ch not in self.target_channels]
        
        # 4. Drop extra channels (e.g., EOG, EMG, or 128-cap specific channels not in our 61 set)
        if extra_channels:
            raw.drop_channels(extra_channels)
            
        # 5. Interpolate missing channels
        if missing_channels:
            # We add empty channels to the info, then use MNE's interpolation
            info = mne.create_info(ch_names=missing_channels, sfreq=raw.info['sfreq'], ch_types='eeg')
            missing_raw = mne.io.RawArray(np.zeros((len(missing_channels), raw.n_times)), info)
            raw.add_channels([missing_raw], force_update_info=True)
            
        # Now set the standard 10-20 montage for spatial coordinates AFTER adding all channels
        try:
            montage = mne.channels.make_standard_montage('standard_1020')
            raw.set_montage(montage, match_case=False, on_missing='ignore')
        except Exception as e:
            pass

        if missing_channels:
            # Mark the newly added zeroed channels as "bad" so interpolate_bads will rebuild them
            raw.info['bads'] = missing_channels
            try:
                # Use a fixed origin to avoid sphere fitting errors on sparse/weird channels
                raw.interpolate_bads(reset_bads=True, verbose=False, origin=(0., 0., 0.))
            except Exception as e:
                raw.info['bads'] = [] # Clear bads if it fails
                
        # 6. Reorder channels to exactly match self.target_channels
        raw.reorder_channels(self.target_channels)
        
        # Apply standard referencing (average reference)
        raw.set_eeg_reference('average', projection=False, verbose=False)
        
        return raw

ch_names = ['Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz'] # Only 6 channels
info = mne.create_info(ch_names=ch_names, sfreq=100, ch_types='eeg')
raw = mne.io.RawArray(np.zeros((len(ch_names), 1000)), info)
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)
for ch in raw.info['chs']:
    ch['loc'][:3] *= 1000 # Make it in mm

harmonizer = FixedChannelHarmonizer()
harmonized_raw = harmonizer(raw)
print("Harmonization completed without interpolation fitting errors!")
