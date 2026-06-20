import json
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def build_manifests(project_root=None):
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    project_root = Path(project_root).resolve()
    jsonl_path = project_root / "data" / "processed" / "manifests" / "detected_files.jsonl"
    out_csv = project_root / "data" / "processed" / "manifests" / "trials.csv"

    if not jsonl_path.exists():
        logger.error(f"Cannot find {jsonl_path}. Run 01_scan_datasets.sh first.")
        return

    rows = []
    
    with open(jsonl_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            dataset = data["dataset"]
            filepath = Path(data["path"])
            
            # Skip non-EEG file types that might have been picked up in the scan (e.g. .yaml, .pt, .json)
            if filepath.suffix.lower() not in ['.edf', '.bdf', '.vhdr', '.mat', '.set']:
                continue
            
                # 1. Chisco Dataset (1 file = 1 trial for .edf files)
            if dataset == "chisco":
                trial_id = filepath.stem
                # Try to parse sub-XX and ses-YY from filename
                # Chisco BIDS filenames: sub-05_ses-06_task-imagine_run-037_eeg.edf
                parts = trial_id.split('_')
                sub = next((p for p in parts if p.startswith("sub-")), "sub-unk")
                ses = next((p for p in parts if p.startswith("ses-")), "ses-unk")
                task = next((p for p in parts if p.startswith("task-")), "task-unk").replace("task-", "")
                run = next((p for p in parts if p.startswith("run-")), "run-unk")
                
                # The run number (e.g., 037) maps to the specific sentence stimulus/category.
                # Use the run number as the initial classification label to enable learning.
                label_val = run.replace("run-", "")
                if label_val.isdigit():
                    label_val = int(label_val)
                
                text_str = ""
                # Try to load the text from the downloaded textdataset
                text_path = project_root / "ds005170-1.1.2" / "textdataset" / f"split_data_{label_val}.xlsx"
                if text_path.exists():
                    try:
                        xlsx_df = pd.read_excel(text_path)
                        # We try to find the longest string in the first row, or column named "Sentence"/"Text"
                        text_str = str(xlsx_df.iloc[0, 0]) # Best effort default
                        for col in xlsx_df.columns:
                            if any(x in str(col).lower() for x in ["text", "sentence", "stimuli", "word"]):
                                text_str = str(xlsx_df[col].iloc[0])
                                break
                    except Exception as e:
                        logger.warning(f"Failed to read {text_path}: {e}")
                
                rows.append({
                    "trial_id": f"chisco_{sub}_{ses}_{run}",
                    "subject_id": sub,
                    "session_id": f"chisco_{sub}_{ses}",
                    "dataset": "chisco",
                    "task": task,
                    "eeg_path": str(filepath),
                    "onset": 0.0,
                    "duration": -1.0,
                    "trial_type": "imagined_speech",
                    "label": label_val,
                    "text": text_str,
                    "raw_metadata": "{}",
                    "split": "train" # Default to train, user can split later
                })
                
            # 2. Thinking Out Loud (Use _events.tsv)
            elif dataset == "thinking_out_loud":
                events_path = filepath.parent / filepath.name.replace("_eeg.bdf", "_events.tsv")
                sub = next((p for p in filepath.stem.split('_') if p.startswith("sub-")), "sub-unk")
                
                if events_path.exists():
                    try:
                        events_df = pd.read_csv(events_path, sep='\t')
                        for idx, ev in events_df.iterrows():
                            onset = float(ev["onset"])
                            duration = float(ev.get("duration", -1.0))
                            trial_type = ev.get("trial_type", "unknown")
                            
                            # Thinking Out Loud has 4 directional commands. 
                            # We extract it from 'value' or 'trial_type' to establish the classification label.
                            label_val = ev.get("value", trial_type)
                            text_str = str(label_val).lower().replace("imagined", "").strip()
                            
                            rows.append({
                                "trial_id": f"tol_{filepath.stem}_trial{idx}",
                                "subject_id": sub,
                                "session_id": f"tol_{sub}",
                                "dataset": "thinking_out_loud",
                                "task": "inner_speech",
                                "eeg_path": str(filepath),
                                "onset": onset,
                                "duration": duration,
                                "trial_type": trial_type,
                                "label": label_val,
                                "text": text_str,
                                "raw_metadata": "{}",
                                "split": "train"
                            })
                    except Exception as e:
                        logger.error(f"Failed to read events for {filepath}: {e}")
                else:
                    # Fallback to single continuous file if no events found
                    rows.append({
                        "trial_id": f"tol_{filepath.stem}",
                        "subject_id": sub,
                        "session_id": f"tol_{sub}",
                        "dataset": "thinking_out_loud",
                        "task": "inner_speech",
                        "eeg_path": str(filepath),
                        "onset": 0.0,
                        "duration": -1.0,
                        "trial_type": "continuous",
                        "text": "",
                        "raw_metadata": "{}",
                        "split": "train"
                    })
                    
            # 3. ZuCo Dataset (.mat files contain sentenceData arrays)
            elif dataset == "zuco":
                try:
                    import h5py
                    with h5py.File(filepath, 'r') as h5_f:
                        if 'sentenceData' in h5_f:
                            # Count number of sentences
                            field = 'rawData' if 'rawData' in h5_f['sentenceData'] else list(h5_f['sentenceData'].keys())[0]
                            num_trials = h5_f['sentenceData'][field].shape[0]
                            
                            sub = filepath.stem.split('_')[-1] if '_' in filepath.stem else "sub-unk"
                            
                            for i in range(num_trials):
                                rows.append({
                                    "trial_id": f"zuco_{filepath.stem}_sen{i}",
                                    "subject_id": sub,
                                    "session_id": f"zuco_{sub}",
                                    "dataset": "zuco",
                                    "task": "natural_reading",
                                    "eeg_path": str(filepath),
                                    "onset": 0.0,
                                    "duration": -1.0,
                                    "trial_type": "sentence",
                                    "label": i,
                                    "text": "", # Zuco text parsing requires H5 deep dive, stub for now
                                    "raw_metadata": "{}",
                                    "split": "train"
                                })
                except Exception as e:
                    logger.warning(f"Failed to parse ZuCo .mat {filepath} for lengths: {e}. Falling back to 1 trial.")
                    rows.append({
                        "trial_id": f"zuco_{filepath.stem}",
                        "subject_id": "sub-unk",
                        "session_id": "zuco_unk",
                        "dataset": "zuco",
                        "task": "natural_reading",
                        "eeg_path": str(filepath),
                        "onset": 0.0,
                        "duration": -1.0,
                        "trial_type": "continuous",
                        "text": "",
                        "raw_metadata": "{}",
                        "split": "train"
                    })

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info(f"Successfully built trials.csv with {len(df)} rows!")

if __name__ == "__main__":
    build_manifests()
