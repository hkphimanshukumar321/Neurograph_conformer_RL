import pandas as pd
import numpy as np
import torch
import mne
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

# Ensure src is in the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.filters import EEGFilter
from src.preprocessing.channel_harmonization import ChannelHarmonizer

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_trial_raw(row) -> mne.io.BaseRaw:
    """
    Loads raw EEG data into an MNE object depending on the dataset.
    """
    dataset = row["dataset"]
    # Handle column naming variations
    path_str = row.get("eeg_path") if pd.notna(row.get("eeg_path")) else row.get("source_file")
    
    if not path_str or pd.isna(path_str):
        raise ValueError(f"No valid file path found in manifest for trial {row['trial_id']}")
        
    path = Path(str(path_str))
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
        
    try:
        if path.suffix == ".edf":
            return mne.io.read_raw_edf(path, preload=True, verbose=False)
        elif path.suffix == ".vhdr":
            return mne.io.read_raw_brainvision(path, preload=True, verbose=False)
        elif path.suffix == ".bdf":
            return mne.io.read_raw_bdf(path, preload=True, verbose=False)
        else:
            raise NotImplementedError(f"Unsupported extension {path.suffix} for dataset {dataset}")
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Offline Preprocessing Cache")
    parser.add_argument("--project_root", type=str, default=".", help="Root directory of the project")
    parser.add_argument("--target_sfreq", type=float, default=250.0, help="Target sampling rate")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    manifest_dir = project_root / "data" / "processed" / "manifests"
    trials_csv_path = manifest_dir / "trials.csv"
    
    if not trials_csv_path.exists():
        logger.error(f"Cannot find {trials_csv_path}. Please run build_manifests.py first.")
        return
        
    df = pd.read_csv(trials_csv_path)
    logger.info(f"Loaded {len(df)} trials from {trials_csv_path}")
    
    mne.set_log_level("WARNING")
    eeg_filter = EEGFilter(target_sfreq=args.target_sfreq, l_freq=0.5, h_freq=100.0, notch_freqs=[50.0, 60.0])
    harmonizer = ChannelHarmonizer()
    
    output_base_dir = project_root / "data" / "processed" / f"common_{int(args.target_sfreq)}hz"
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    fail_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Caching trials"):
        dataset = row["dataset"]
        trial_id = row["trial_id"]
        
        dataset_out_dir = output_base_dir / dataset
        dataset_out_dir.mkdir(parents=True, exist_ok=True)
        
        out_file = dataset_out_dir / f"{trial_id}.pt"
        if out_file.exists():
            success_count += 1
            continue
            
        try:
            # 1. Load Raw data
            raw = load_trial_raw(row)
            
            # 2. Slice to trial boundaries if onset/duration exist
            onset = float(row.get("onset", 0.0))
            duration = float(row.get("duration", -1.0))
            
            if duration != -1.0 and duration > 0:
                # Crop MNE raw object
                tmax = min(onset + duration, raw.times[-1])
                raw.crop(tmin=onset, tmax=tmax)
                
            # 3. Apply Preprocessing Harmonization
            raw = eeg_filter(raw)
            raw = harmonizer(raw)
            
            # 4. Convert to Tensor and Save
            # Shape should be (61, T)
            tensor_data = torch.tensor(raw.get_data(), dtype=torch.float32)
            torch.save(tensor_data, out_file)
            
            success_count += 1
            
        except NotImplementedError as e:
            logger.debug(str(e))
            fail_count += 1
        except Exception as e:
            logger.debug(f"Failed trial {trial_id}: {e}")
            fail_count += 1
            
    logger.info(f"Finished caching! Success: {success_count}, Failed: {fail_count}")

if __name__ == "__main__":
    main()
