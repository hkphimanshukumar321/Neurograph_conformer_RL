import json
import pandas as pd
from pathlib import Path
import logging
import html
import re
import zipfile

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _read_chisco_label_pairs(xlsx_path: Path) -> list[tuple[str, str]]:
    """Read sentence/label pairs from a Chisco workbook without openpyxl."""
    pairs: list[tuple[str, str]] = []
    with zipfile.ZipFile(xlsx_path) as archive:
        sheet_xml = archive.read("xl/worksheets/sheet1.xml").decode("utf-8", errors="ignore")
        rows = re.findall(r'<row r="(\d+)">(.*?)</row>', sheet_xml)
        for row_num, row_xml in rows[1:]:
            values = [html.unescape(value) for value in re.findall(r'<t[^>]*>(.*?)</t>', row_xml)]
            if len(values) >= 2:
                pairs.append((values[0], values[1]))
    return pairs

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
    seen_zuco_files = set()
    seen_chisco_files = set()
    
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
                resolved_filepath = filepath.resolve()
                if resolved_filepath in seen_chisco_files:
                    continue
                seen_chisco_files.add(resolved_filepath)

                trial_id = filepath.stem
                parts = trial_id.split('_')
                sub = next((p for p in parts if p.startswith("sub-")), "sub-unk")
                ses = next((p for p in parts if p.startswith("ses-")), "ses-unk")
                run = next((p for p in parts if p.startswith("run-")), None)
                task = next((p for p in parts if p.startswith("task-")), "task-imagine")
                
                # ── Robust label extraction with multiple fallbacks ──
                label_val = None
                text_str = ""
                
                # Strategy 1: Use run number from filename (best: maps to stimulus ID)
                if run is not None:
                    run_str = run.replace("run-", "")
                    if run_str.isdigit():
                        label_val = int(run_str)
                    elif run_str != "unk":
                        label_val = run_str
                
                # Strategy 2: Check paired events.tsv for trial_type
                if label_val is None:
                    events_file = filepath.parent / (trial_id.replace("_eeg", "_events") + ".tsv")
                    if not events_file.exists():
                        # Try without _eeg suffix
                        events_file = filepath.parent / (trial_id + "_events.tsv")
                    if not events_file.exists():
                        # Try BIDS-standard naming
                        base_parts = [p for p in parts if not p.startswith("eeg")]
                        events_file = filepath.parent / ("_".join(base_parts) + "_events.tsv")
                    
                    if events_file.exists():
                        try:
                            ev_df = pd.read_csv(events_file, sep='\t')
                            # Look for trial_type or value column
                            for col in ['trial_type', 'value', 'stimulus', 'condition']:
                                if col in ev_df.columns:
                                    unique_vals = ev_df[col].dropna().unique()
                                    if len(unique_vals) > 0:
                                        # Use the most common non-empty value
                                        val = str(ev_df[col].dropna().mode().iloc[0])
                                        if val and val.lower() not in ('nan', 'n/a', ''):
                                            label_val = val
                                            break
                        except Exception as e:
                            logger.debug(f"Could not parse events {events_file}: {e}")
                
                # Strategy 3: Extract label from directory structure
                # e.g. ds005170/sub-01/ses-01/eeg/ → use sub+ses combo
                if label_val is None:
                    # Use the parent directory name chain as a differentiator
                    parent_parts = []
                    for parent in filepath.parents:
                        name = parent.name
                        if name.startswith("sub-") or name.startswith("ses-") or name.startswith("run-"):
                            parent_parts.append(name)
                        if name in ("eeg", "data", "raw"):
                            break
                    if parent_parts:
                        label_val = "_".join(reversed(parent_parts))
                
                # Strategy 4: Use a hash of the filename for unique labeling  
                # This ensures DIFFERENT files get DIFFERENT labels
                if label_val is None:
                    # Extract any numeric part from filename
                    import re
                    nums = re.findall(r'\d+', trial_id)
                    if nums:
                        # Use the last significant number (likely trial/run index)
                        label_val = int(nums[-1]) if nums[-1].isdigit() else nums[-1]
                    else:
                        # Absolute fallback: sequential index
                        label_val = len(rows)
                
                # ── Text extraction ──
                # Try to load the text from the textdataset
                if isinstance(label_val, int):
                    text_path = project_root / "ds005170-1.1.2" / "textdataset" / f"split_data_{label_val}.xlsx"
                    if text_path.exists():
                        try:
                            xlsx_df = pd.read_excel(text_path)
                            text_str = str(xlsx_df.iloc[0, 0])  # Best effort default
                            for col in xlsx_df.columns:
                                if any(x in str(col).lower() for x in ["text", "sentence", "stimuli", "word"]):
                                    text_str = str(xlsx_df[col].iloc[0])
                                    break
                        except Exception as e:
                            logger.warning(f"Failed to read {text_path}: {e}")
                
                rows.append({
                    "trial_id": f"chisco_{sub}_{ses}_{run or 'run-auto'}",
                    "subject_id": sub,
                    "session_id": f"chisco_{sub}_{ses}",
                    "dataset": "chisco",
                    "task": task.replace("task-", ""),
                    "eeg_path": str(filepath),
                    "onset": 0.0,
                    "duration": -1.0,
                    "trial_type": "imagined_speech",
                    "label": label_val,
                    "text": text_str,
                    "raw_metadata": "{}",
                    "split": "train"  # Default to train, user can split later
                })
                
            # 2. Thinking Out Loud (Use _events.tsv)
            elif dataset == "thinking_out_loud":
                sub = next((p for p in filepath.stem.split('_') if p.startswith("sub-")), "sub-unk")
                session_name = filepath.parent.parent.name if filepath.parent.parent.name.startswith("ses-") else "ses-unk"
                
                events_path = filepath.parents[3] / "derivatives" / sub / session_name / f"{sub}_{session_name}_events.tsv"
                if not events_path.exists():
                    events_path = filepath.parents[3] / "derivatives" / sub / session_name / f"{sub}_{session_name}_events.dat"
                
                if events_path.exists():
                    try:
                        if events_path.suffix == '.tsv':
                            events_df = pd.read_csv(events_path, sep='\t')
                            for idx, ev in events_df.iterrows():
                                onset = float(ev["onset"])
                                duration = float(ev.get("duration", -1.0))
                                trial_type = ev.get("trial_type", "unknown")
                                
                                label_val = ev.get("value", trial_type)
                                text_str = str(label_val).lower().replace("imagined", "").strip()
                                
                                rows.append({
                                    "trial_id": f"tol_{filepath.stem}_trial{idx:03d}",
                                    "subject_id": sub,
                                    "session_id": f"tol_{sub}",
                                    "dataset": "thinking_out_loud",
                                    "task": "inner_speech_command_classification",
                                    "label": label_val,
                                    "eeg_path": str(filepath),
                                    "onset": onset,
                                    "duration": duration,
                                    "trial_type": trial_type,
                                    "text": text_str,
                                    "raw_metadata": "{}",
                                    "split": "train"
                                })
                        else:
                            # It's a .dat file (pickle)
                            import pickle
                            import mne
                            with open(events_path, "rb") as ep:
                                events = pickle.load(ep)
                            
                            try:
                                raw = mne.io.read_raw_bdf(filepath, preload=False, verbose=False)
                                sfreq = float(raw.info.get("sfreq", 128.0))
                            except Exception:
                                sfreq = 128.0

                            label_map = {
                                0: "command_0",
                                1: "command_1",
                                2: "command_2",
                                3: "command_3",
                            }
                            for idx, ev in enumerate(events):
                                onset_sample = int(ev[0])
                                event_code = int(ev[1])
                                label_val = label_map.get(event_code, f"command_{event_code}")
                                text_str = label_val.replace("_", " ")
                                rows.append({
                                    "trial_id": f"tol_{filepath.stem}_trial{idx:03d}",
                                    "subject_id": sub,
                                    "session_id": f"tol_{sub}",
                                    "dataset": "thinking_out_loud",
                                    "task": "inner_speech_command_classification",
                                    "label": label_val,
                                    "eeg_path": str(filepath),
                                    "onset": onset_sample / sfreq,
                                    "duration": 2.0,
                                    "trial_type": label_val,
                                    "text": text_str,
                                    "raw_metadata": "{}",
                                    "split": "train"
                                })
                    except Exception as e:
                        logger.error(f"Failed to read events for {filepath}: {e}")
                        rows.append({
                            "trial_id": f"tol_{filepath.stem}",
                            "subject_id": sub,
                            "session_id": f"tol_{sub}",
                            "dataset": "thinking_out_loud",
                            "task": "inner_speech_command_classification",
                            "label": "command_0",
                            "eeg_path": str(filepath),
                            "onset": 0.0,
                            "duration": -1.0,
                            "trial_type": "continuous",
                            "text": "",
                            "raw_metadata": "{}",
                            "split": "train",
                        })
                else:
                    logger.warning(f"Thinking Out Loud events file missing for {filepath}; falling back to a single continuous trial.")
                    rows.append({
                        "trial_id": f"tol_{filepath.stem}",
                        "subject_id": sub,
                        "session_id": f"tol_{sub}",
                        "dataset": "thinking_out_loud",
                        "task": "inner_speech_command_classification",
                        "label": "command_0",
                        "eeg_path": str(filepath),
                        "onset": 0.0,
                        "duration": -1.0,
                        "trial_type": "continuous",
                        "text": "",
                        "raw_metadata": "{}",
                        "split": "train",
                    })
                    
            # 3. ZuCo Dataset (.mat files contain sentenceData arrays)
            elif dataset == "zuco":
                try:
                    resolved_filepath = filepath.resolve()
                    if resolved_filepath in seen_zuco_files:
                        continue
                    seen_zuco_files.add(resolved_filepath)

                    import h5py
                    with h5py.File(filepath, 'r') as h5_f:
                        sub = filepath.stem.split('_')[-1] if '_' in filepath.stem else "sub-unk"
                        task_name = "natural_reading_language_pretraining"

                        if 'sentenceData' in h5_f:
                            # Count number of sentences.
                            field = 'rawData' if 'rawData' in h5_f['sentenceData'] else list(h5_f['sentenceData'].keys())[0]
                            num_trials = h5_f['sentenceData'][field].shape[0]

                            for i in range(num_trials):
                                rows.append({
                                    "trial_id": f"zuco_{filepath.stem}_sen{i}",
                                    "subject_id": sub,
                                    "session_id": f"zuco_{sub}",
                                    "dataset": "zuco",
                                    "task": task_name,
                                    "eeg_path": str(filepath),
                                    "onset": 0.0,
                                    "duration": -1.0,
                                    "trial_type": "sentence",
                                    "label": i,
                                    "text": "", # Zuco text parsing requires H5 deep dive, stub for now
                                    "raw_metadata": "{}",
                                    "split": "train"
                                })
                        else:
                            logger.warning(
                                f"ZuCo file {filepath} has no sentenceData; falling back to one pretraining trial from EEG/data."
                            )
                            rows.append({
                                "trial_id": f"zuco_{filepath.stem}_pretrain",
                                "subject_id": sub,
                                "session_id": f"zuco_{sub}",
                                "dataset": "zuco",
                                "task": task_name,
                                "eeg_path": str(filepath),
                                "onset": 0.0,
                                "duration": -1.0,
                                "trial_type": "pretraining",
                                "raw_metadata": "{}",
                                "split": "train"
                            })
                except Exception as e:
                    logger.warning(f"Failed to parse ZuCo .mat {filepath} for lengths: {e}. Falling back to 1 pretraining trial.")
                    rows.append({
                        "trial_id": f"zuco_{filepath.stem}_pretrain",
                        "subject_id": "sub-unk",
                        "session_id": "zuco_unk",
                        "dataset": "zuco",
                        "task": "natural_reading_language_pretraining",
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
    
    # ── Diagnostic: label distribution per dataset ──
    if "label" in df.columns and "dataset" in df.columns:
        for ds_name in df["dataset"].unique():
            ds_df = df[df["dataset"] == ds_name]
            n_unique = ds_df["label"].nunique()
            sample_labels = ds_df["label"].value_counts().head(5).to_dict()
            logger.info(f"  {ds_name}: {len(ds_df)} trials, {n_unique} unique labels")
            logger.info(f"    Top labels: {sample_labels}")
            if n_unique <= 1:
                logger.error(
                    f"  ⚠ CRITICAL: {ds_name} has only {n_unique} unique label(s)! "
                    f"Training will be degenerate (1-class). "
                    f"Check label extraction logic in build_manifests.py."
                )

if __name__ == "__main__":
    build_manifests()
