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
                run = next((p for p in parts if p.startswith("run-")), "run-unk")
<<<<<<< HEAD

                subject_token = sub.split("-", 1)[-1]
                try:
                    subject_index = int(subject_token)
                except ValueError:
                    subject_index = None

                workbook_path = filepath.parents[3] / "textdataset" / f"split_data_{subject_index}.xlsx" if subject_index is not None else None

                try:
                    import mne

                    raw = mne.io.read_raw_edf(filepath, preload=False, verbose=False)
                    onsets = list(raw.annotations.onset)
                except Exception as e:
                    logger.warning(f"Failed to read Chisco annotations for {filepath}: {e}. Falling back to a single labeled trial.")
                    onsets = []

                label_pairs: list[tuple[str, str]] = []
                if workbook_path is not None and workbook_path.exists():
                    try:
                        label_pairs = _read_chisco_label_pairs(workbook_path)
                    except Exception as e:
                        logger.warning(f"Failed to read Chisco workbook {workbook_path}: {e}")

                if onsets and label_pairs:
                    num_trials = min(len(onsets), len(label_pairs))
                    if len(onsets) != len(label_pairs):
                        logger.warning(
                            f"Chisco file {filepath} has {len(onsets)} annotations but {len(label_pairs)} workbook rows; using {num_trials}."
                        )

                    for i in range(num_trials):
                        sentence_text, label_text = label_pairs[i]
                        rows.append({
                            "trial_id": f"chisco_{sub}_{ses}_{run}_trial{i:03d}",
                            "subject_id": sub,
                            "session_id": f"chisco_{sub}_{ses}",
                            "dataset": "chisco",
                            "task": "sentence_level_imagined_speech",
                            "label": label_text,
                            "eeg_path": str(filepath),
                            "onset": float(onsets[i]),
                            "duration": 2.0,
                            "trial_type": sentence_text,
                            "raw_metadata": json.dumps({"sentence": sentence_text, "label": label_text}, ensure_ascii=False),
                            "split": "train",
                        })
                else:
                    rows.append({
                        "trial_id": f"chisco_{sub}_{ses}_{run}",
                        "subject_id": sub,
                        "session_id": f"chisco_{sub}_{ses}",
                        "dataset": "chisco",
                        "task": "sentence_level_imagined_speech",
                        "label": "imagined_speech",
                        "eeg_path": str(filepath),
                        "onset": 0.0,
                        "duration": 2.0,
                        "trial_type": "imagined_speech",
                        "raw_metadata": "{}",
                        "split": "train",
                    })
=======
                
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
>>>>>>> c3493edfeb3979b79d52d844a407b49202bcc020
                
            # 2. Thinking Out Loud (Use _events.tsv)
            elif dataset == "thinking_out_loud":
                sub = next((p for p in filepath.stem.split('_') if p.startswith("sub-")), "sub-unk")
<<<<<<< HEAD
                session_name = filepath.parent.parent.name if filepath.parent.parent.name.startswith("ses-") else "ses-unk"
                events_path = filepath.parents[3] / "derivatives" / sub / session_name / f"{sub}_{session_name}_events.dat"

                try:
                    import pickle
                    import mne

                    raw = mne.io.read_raw_bdf(filepath, preload=False, verbose=False)
                    sfreq = float(raw.info.get("sfreq", 128.0))

                    if events_path.exists():
                        events = pickle.load(open(events_path, "rb"))
                        label_map = {
                            0: "command_0",
                            1: "command_1",
                            2: "command_2",
                            3: "command_3",
                        }

                        for idx, ev in enumerate(events):
                            onset_sample = int(ev[0])
                            event_code = int(ev[1])
=======
                
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
                            
>>>>>>> c3493edfeb3979b79d52d844a407b49202bcc020
                            rows.append({
                                "trial_id": f"tol_{filepath.stem}_trial{idx:03d}",
                                "subject_id": sub,
                                "session_id": f"tol_{sub}",
                                "dataset": "thinking_out_loud",
                                "task": "inner_speech_command_classification",
                                "label": label_map.get(event_code, f"command_{event_code}"),
                                "eeg_path": str(filepath),
<<<<<<< HEAD
                                "onset": onset_sample / sfreq,
                                "duration": 2.0,
                                "trial_type": label_map.get(event_code, f"command_{event_code}"),
                                "raw_metadata": json.dumps({"event_code": event_code}, ensure_ascii=False),
                                "split": "train",
=======
                                "onset": onset,
                                "duration": duration,
                                "trial_type": trial_type,
                                "label": label_val,
                                "text": text_str,
                                "raw_metadata": "{}",
                                "split": "train"
>>>>>>> c3493edfeb3979b79d52d844a407b49202bcc020
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
                            "duration": 2.0,
                            "trial_type": "command_0",
                            "raw_metadata": "{}",
                            "split": "train",
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
<<<<<<< HEAD
                        "duration": 2.0,
                        "trial_type": "command_0",
=======
                        "duration": -1.0,
                        "trial_type": "continuous",
                        "text": "",
>>>>>>> c3493edfeb3979b79d52d844a407b49202bcc020
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
<<<<<<< HEAD
                        "trial_type": "pretraining",
=======
                        "trial_type": "continuous",
                        "text": "",
>>>>>>> c3493edfeb3979b79d52d844a407b49202bcc020
                        "raw_metadata": "{}",
                        "split": "train"
                    })

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info(f"Successfully built trials.csv with {len(df)} rows!")

if __name__ == "__main__":
    build_manifests()
