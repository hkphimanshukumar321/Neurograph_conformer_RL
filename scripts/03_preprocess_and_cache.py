import pandas as pd
import numpy as np
import torch
import mne
import logging
import argparse
import sys
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

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

def process_group(source_file, group, output_base_dir, target_sfreq):
    """Worker function to process one continuous EEG file and extract its trials."""
    mne.set_log_level("WARNING")
    # Instantiate processors inside the worker to avoid pickling issues
    eeg_filter = EEGFilter(target_sfreq=target_sfreq, l_freq=0.5, h_freq=100.0, notch_freqs=[50.0, 60.0])
    harmonizer = ChannelHarmonizer()
    
    success = 0
    fail = 0
    skip = 0
    
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
        return 0, len(group), 0
        
    # --- ZUCO .mat DATASET HANDLING ---
    if dataset == "zuco":
        try:
            import h5py
            with h5py.File(source_file, 'r') as f:
                if 'sentenceData' not in f:
                    raise ValueError(f"ZuCo file {source_file} missing 'sentenceData' struct")
                
                sentence_data = f['sentenceData']
                
                for idx_in_group, (idx, row) in enumerate(group.iterrows()):
                    trial_id = row["trial_id"]
                    out_file = dataset_out_dir / f"{trial_id}.pt"
                    
                    if out_file.exists():
                        skip += 1
                        continue
                        
                    try:
                        field_to_extract = 'rawData' if 'rawData' in sentence_data else list(sentence_data.keys())[0]
                        ref = sentence_data[field_to_extract][idx_in_group, 0]
                        eeg_data = np.array(f[ref])
                        
                        if eeg_data.shape[0] > eeg_data.shape[1]:
                            eeg_data = eeg_data.T
                            
                        tensor_data = torch.tensor(eeg_data, dtype=torch.float32)
                        torch.save(tensor_data, out_file)
                        success += 1
                    except Exception as e:
                        fail += 1
        except Exception as e:
            fail += len(group)
        return success, skip, fail

    # --- STANDARD CONTINUOUS FILES (Chisco, Thinking Out Loud) ---
    try:
        raw_continuous = load_continuous_raw(source_file, dataset)
        raw_continuous = eeg_filter(raw_continuous)
        raw_continuous = harmonizer(raw_continuous)
        
        for idx, row in group.iterrows():
            trial_id = row["trial_id"]
            out_file = dataset_out_dir / f"{trial_id}.pt"
            
            if out_file.exists():
                skip += 1
                continue
                
            onset = float(row.get("onset", row.get("start_sec", 0.0)))
            duration = float(row.get("duration", -1.0))
            
            if dataset == "thinking_out_loud" and onset > raw_continuous.times[-1]:
                bdf_path = Path(source_file)
                events_path = bdf_path.parent / bdf_path.name.replace("_eeg.bdf", "_events.tsv")
                
                if events_path.exists():
                    events_df = pd.read_csv(events_path, sep="\t")
                    idx_in_group = group.index.get_loc(idx)
                    if idx_in_group < len(events_df):
                        onset = float(events_df.iloc[idx_in_group]["onset"])
                        dur_val = events_df.iloc[idx_in_group].get("duration")
                        if pd.notna(dur_val):
                            duration = float(dur_val)
            
            trial_raw = raw_continuous.copy()
            if duration != -1.0 and duration > 0:
                tmax = min(onset + duration, trial_raw.times[-1])
                trial_raw.crop(tmin=onset, tmax=tmax)
                
            tensor_data = torch.tensor(trial_raw.get_data(), dtype=torch.float32)
            torch.save(tensor_data, out_file)
            success += 1
            
    except NotImplementedError:
        fail += len(group)
    except Exception as e:
        fail += len(group)
        
    return success, skip, fail

def main():
    parser = argparse.ArgumentParser(description="Offline Preprocessing Cache (Parallel)")
    parser.add_argument("--project_root", type=str, default=".", help="Root directory of the project")
    parser.add_argument("--target_sfreq", type=float, default=250.0, help="Target sampling rate")
    parser.add_argument("--jobs", type=int, default=-1, help="Number of parallel jobs (-1 for max_cores - 1). Use lower values to prevent RAM OOM.")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    manifest_dir = project_root / "data" / "processed" / "manifests"
    trials_csv_path = manifest_dir / "trials.csv"
    
    if not trials_csv_path.exists():
        logger.error(f"Cannot find {trials_csv_path}. Please run 02_prepare_metadata.sh first.")
        return
        
    df = pd.read_csv(trials_csv_path)
    logger.info(f"Loaded {len(df)} trials from {trials_csv_path}")
    
    output_base_dir = project_root / "data" / "processed" / f"common_{int(args.target_sfreq)}hz"
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    source_col = "eeg_path" if "eeg_path" in df.columns else "source_file"
    grouped = df.groupby(source_col)
    logger.info(f"Grouped {len(df)} trials into {len(grouped)} unique continuous recording files.")
    
    # -- OOM GUARDRAIL LOGIC --
    # CPU preprocessing can use ~4-8GB RAM per worker for massive EEG files.
    max_cores = multiprocessing.cpu_count()
    if args.jobs == -1:
        # Default guardrail: use half the available cores to prevent system memory from hitting 100%
        n_jobs = max(1, max_cores // 2) 
        logger.info(f"Auto-configured n_jobs={n_jobs} (of {max_cores} total cores) as an OOM guardrail.")
    else:
        n_jobs = min(args.jobs, max_cores)
        logger.info(f"Using manual n_jobs={n_jobs}.")

    # Execute in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_group)(source_file, group, output_base_dir, args.target_sfreq)
        for source_file, group in tqdm(grouped, desc=f"Parallel Processing ({n_jobs} workers)")
    )
    
    total_success = sum(r[0] for r in results)
    total_skip = sum(r[1] for r in results)
    total_fail = sum(r[2] for r in results)
            
    logger.info(f"Finished parallel caching! Success: {total_success}, Skipped: {total_skip}, Failed: {total_fail}")

if __name__ == "__main__":
    main()
