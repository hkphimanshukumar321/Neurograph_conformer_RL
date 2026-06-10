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

def load_continuous_raw(path_str: str, dataset: str) -> mne.io.BaseRaw:
    """
    Loads raw EEG data into an MNE object depending on the dataset.
    """
    if not path_str or pd.isna(path_str):
        raise ValueError("No valid file path found in manifest")
        
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
    skip_count = 0
    
    source_col = "eeg_path" if "eeg_path" in df.columns else "source_file"
    grouped = df.groupby(source_col)
    
    logger.info(f"Grouped 5898 trials into {len(grouped)} unique continuous recording files.")
    
    for source_file, group in tqdm(grouped, desc="Processing continuous files", total=len(grouped)):
        dataset = group.iloc[0]["dataset"]
        dataset_out_dir = output_base_dir / dataset
        dataset_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if all trials for this file are already cached
        all_cached = True
        for _, row in group.iterrows():
            if not (dataset_out_dir / f"{row['trial_id']}.pt").exists():
                all_cached = False
                break
                
        if all_cached:
            skip_count += len(group)
            continue
            
        try:
            # 1. Load the continuous raw file ONCE into memory
            raw_continuous = load_continuous_raw(source_file, dataset)
            
            # 2. Filter & Harmonize the entire continuous recording ONCE
            raw_continuous = eeg_filter(raw_continuous)
            raw_continuous = harmonizer(raw_continuous)
            
            # 3. Extract and save all individual trials for this recording
            for idx, row in group.iterrows():
                trial_id = row["trial_id"]
                out_file = dataset_out_dir / f"{trial_id}.pt"
                
                if out_file.exists():
                    skip_count += 1
                    continue
                    
                # Slice trial bounds
                onset = float(row.get("onset", row.get("start_sec", 0.0)))
                duration = float(row.get("duration", -1.0))
                
                # We must use copy() to avoid altering the continuous object
                trial_raw = raw_continuous.copy()
                
                if duration != -1.0 and duration > 0:
                    tmax = min(onset + duration, trial_raw.times[-1])
                    trial_raw.crop(tmin=onset, tmax=tmax)
                    
                # Convert to Tensor and Save (61, T)
                tensor_data = torch.tensor(trial_raw.get_data(), dtype=torch.float32)
                torch.save(tensor_data, out_file)
                
                success_count += 1
                
        except NotImplementedError as e:
            logger.debug(str(e))
            fail_count += len(group)
        except Exception as e:
            logger.error(f"Failed processing file {source_file}: {e}")
            fail_count += len(group)
            
    logger.info(f"Finished caching! Success: {success_count}, Skipped: {skip_count}, Failed: {fail_count}")

if __name__ == "__main__":
    main()
