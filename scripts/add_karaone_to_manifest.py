import pandas as pd
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Append KaraOne files to trials.csv manifest")
    parser.add_argument("--karaone_dir", type=str, required=True, help="Absolute path to KaraOne dataset on your server (e.g. /home/laksanthd/data/karaone)")
    parser.add_argument("--manifest_path", type=str, default="data/processed/manifests/trials.csv")
    args = parser.parse_args()
    
    karaone_dir = Path(args.karaone_dir)
    manifest_path = Path(args.manifest_path)
    
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found at {manifest_path}")
        return
        
    if not karaone_dir.exists():
        print(f"[ERROR] KaraOne directory {karaone_dir} does not exist.")
        return

    # Find all KaraOne files (usually .mat or .edf)
    files = list(karaone_dir.rglob("*.mat")) + list(karaone_dir.rglob("*.edf"))
    
    if not files:
        print(f"[ERROR] No .mat or .edf files found in {karaone_dir}")
        return
        
    df = pd.read_csv(manifest_path)
    
    # Check if kara_one is already there to prevent duplicates
    if "kara_one" in df["dataset"].unique() or "karaone" in df["dataset"].unique():
        print("[WARNING] KaraOne already seems to be in the manifest. Skipping to prevent duplicates.")
        return

    new_rows = []
    print(f"Found {len(files)} files in KaraOne directory.")
    
    for idx, f in enumerate(files):
        # We create a mapping for the caching script. 
        # For KaraOne, we set start_sec=0 and end_sec=-1 to load the whole file/epochs.
        row = {
            "dataset": "kara_one",
            "subject_id": f.stem.split('_')[0],  # e.g. MM05
            "session_id": "ses-01",
            "trial_id": f"kara_one_{f.stem}_{idx}",
            "condition": "unknown",
            "label_text": "unknown",
            "label_type": "unknown",
            "start_sample": 0,
            "end_sample": -1,
            "start_sec": 0.0,
            "end_sec": -1.0,
            "split": "train",
            "source_file": str(f.resolve())
        }
        new_rows.append(row)
        
    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([df, new_df], ignore_index=True)
    combined.to_csv(manifest_path, index=False)
    
    print(f"[SUCCESS] Appended {len(new_rows)} KaraOne files to {manifest_path}")
    print("You can now re-run scripts/04_cache_preprocessed_data.py to cache them.")

if __name__ == "__main__":
    main()
