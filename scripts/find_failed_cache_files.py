"""
Script to find out exactly which EEG files failed during caching and why.
Run this on your server where the data is located.

Usage:
    python scripts/find_failed_cache_files.py --project_root .
"""

import pandas as pd
import mne
import argparse
from pathlib import Path
from tqdm import tqdm
import traceback

def load_continuous_raw(path_str: str) -> mne.io.BaseRaw:
    path = Path(str(path_str))
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
        
    if path.suffix == ".edf":
        return mne.io.read_raw_edf(path, preload=True, verbose=False)
    elif path.suffix == ".vhdr":
        return mne.io.read_raw_brainvision(path, preload=True, verbose=False)
    elif path.suffix == ".bdf":
        return mne.io.read_raw_bdf(path, preload=True, verbose=False)
    else:
        raise NotImplementedError(f"Unsupported extension {path.suffix}")

def main():
    parser = argparse.ArgumentParser(description="Find failed cache files")
    parser.add_argument("--project_root", type=str, default=".", help="Root directory of the project")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    manifest_dir = project_root / "data" / "processed" / "manifests"
    trials_csv_path = manifest_dir / "trials.csv"
    
    if not trials_csv_path.exists():
        print(f"[ERROR] Cannot find {trials_csv_path}")
        return
        
    df = pd.read_csv(trials_csv_path)
    print(f"[INFO] Loaded {len(df)} trials.")
    
    mne.set_log_level("WARNING")
    
    source_col = "eeg_path" if "eeg_path" in df.columns else "source_file"
    grouped = df.groupby(source_col)
    
    failed_files = []
    
    print(f"\n[INFO] Scanning {len(grouped)} unique recording files...\n")
    
    for source_file, group in tqdm(grouped, desc="Testing files", total=len(grouped)):
        dataset = group.iloc[0]["dataset"]
        try:
            # Test 1: Can we load it?
            raw = load_continuous_raw(source_file)
            
            # Test 2: Can we crop all expected trials?
            file_max_time = raw.times[-1] if len(raw.times) > 0 else 0
            for idx, row in group.iterrows():
                onset = float(row.get("onset", row.get("start_sec", 0.0)))
                duration = float(row.get("duration", -1.0))
                
                # Check if onset is beyond file length
                if onset > file_max_time:
                    raise ValueError(
                        f"Trial onset ({onset}s) is beyond file length ({file_max_time}s). "
                        "The recording may be truncated."
                    )
                    
        except Exception as e:
            failed_files.append({
                "dataset": dataset,
                "file": source_file,
                "error": str(e),
                "type": type(e).__name__
            })

    print("\n" + "="*60)
    print(f"FOUND {len(failed_files)} FAILED FILES")
    print("="*60)
    
    for idx, f in enumerate(failed_files, 1):
        print(f"\n--- Failure {idx} ---")
        print(f"Dataset: {f['dataset']}")
        print(f"File:    {f['file']}")
        print(f"Error:   [{f['type']}] {f['error']}")

    # Save report
    if failed_files:
        report_df = pd.DataFrame(failed_files)
        report_path = project_root / "failed_cache_report.csv"
        report_df.to_csv(report_path, index=False)
        print(f"\n[INFO] Saved detailed failure report to {report_path}")

if __name__ == "__main__":
    main()
