import numpy as np
import mne
import logging

logger = logging.getLogger(__name__)

# Standard 10-20 layout subset (62 channels common in most high-density caps)
STANDARD_CHANNELS = [
    'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
    'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz',
    'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4',
    'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz',
    'PO4', 'PO8', 'O1', 'Oz', 'O2'
] # Length 61

class ChannelHarmonizer:
    """
    Harmonizes EEG channels across different datasets to a consistent 10-20 layout.
    Drops extra channels, and uses spherical spline interpolation for missing channels.
    """
    def __init__(self, target_channels=None):
        self.target_channels = target_channels if target_channels is not None else STANDARD_CHANNELS
        
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
        
        # 2. Set standard 10-20 montage for spatial coordinates (needed for interpolation)
        try:
            montage = mne.channels.make_standard_montage('standard_1020')
            raw.set_montage(montage, match_case=False, on_missing='ignore')
        except Exception as e:
            logger.warning(f"Could not set standard montage: {e}")
            
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
            
            # Set montage for the missing raw to give it coordinates
            try:
                missing_raw.set_montage(montage, match_case=False, on_missing='ignore')
            except:
                pass
                
            raw.add_channels([missing_raw], force_update_info=True)
            
            # Mark the newly added zeroed channels as "bad" so interpolate_bads will rebuild them
            raw.info['bads'] = missing_channels
            try:
                raw.interpolate_bads(reset_bads=True, verbose=False)
            except Exception as e:
                logger.warning(f"Interpolation failed: {e}. Channels {missing_channels} will remain zeros.")
                raw.info['bads'] = [] # Clear bads if it fails
                
        # 6. Reorder channels to exactly match self.target_channels
        raw.reorder_channels(self.target_channels)
        
        # Apply standard referencing (average reference)
        raw.set_eeg_reference('average', projection=False, verbose=False)
        
        return raw
