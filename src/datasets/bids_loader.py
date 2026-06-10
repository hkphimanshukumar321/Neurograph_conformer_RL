import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def parse_bids_dataset(dataset_name: str, root_path: str):
    """
    Parses a BIDS-compliant dataset and extracts subjects, sessions, and trials.
    
    Returns
    -------
    subjects_df : pd.DataFrame
    sessions_df : pd.DataFrame
    trials_df : pd.DataFrame
    """
    root_path = Path(root_path)
    if not root_path.exists():
        logger.warning(f"BIDS root {root_path} does not exist.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    subjects = []
    sessions = []
    trials = []
    
    # 1. Parse participants.tsv
    participants_file = root_path / "participants.tsv"
    if participants_file.exists():
        try:
            df_part = pd.read_csv(participants_file, sep='\t')
            for _, row in df_part.iterrows():
                sub_id = row.get("participant_id", "")
                if pd.isna(sub_id): continue
                subjects.append({
                    "subject_id": sub_id,
                    "dataset": dataset_name,
                    "age": row.get("age", "n/a"),
                    "sex": row.get("sex", "n/a"),
                })
        except Exception as e:
            logger.error(f"Error reading {participants_file}: {e}")
    else:
        # Fallback: scan for sub-* directories
        for sub_dir in root_path.glob("sub-*"):
            if sub_dir.is_dir():
                subjects.append({
                    "subject_id": sub_dir.name,
                    "dataset": dataset_name,
                    "age": "n/a",
                    "sex": "n/a"
                })
                
    # 2. Parse sessions & trials by globbing *_events.tsv or events.tsv
    events_files = list(root_path.rglob("*events.tsv")) + list(root_path.rglob("*events.csv"))
    
    if not events_files:
        logger.warning(f"No events files found in {root_path}! Cannot generate trials.")
        
    for ev_file in events_files:
        # e.g. sub-01_ses-01_task-inner_run-01_events.tsv
        ev_name = ev_file.name.replace("_events.tsv", "").replace("events.tsv", "").replace("_events.csv", "").replace("events.csv", "")
        if ev_name.endswith("_"): ev_name = ev_name[:-1]
        
        parts = ev_name.split("_")
        
        sub_id = None
        ses_id = "ses-01"
        task = "unknown"
        run = "01"
        
        for p in parts:
            if p.startswith("sub-"): sub_id = p
            elif p.startswith("ses-"): ses_id = p
            elif p.startswith("task-"): task = p.split("-", 1)[1]
            elif p.startswith("run-"): run = p.split("-", 1)[1]
            
        if not sub_id:
            # Fallback if filename lacks sub-, try to get it from parent directory name
            if "sub-" in ev_file.parent.name:
                sub_id = ev_file.parent.name
            elif "sub-" in ev_file.parent.parent.name:
                sub_id = ev_file.parent.parent.name
            else:
                continue
            
        session_uid = f"{dataset_name}_{sub_id}_{ses_id}"
        
        sessions.append({
            "session_id": session_uid,
            "subject_id": sub_id,
            "dataset": dataset_name,
            "task": task
        })
        
        # 3. Find matching EEG file
        eeg_dir = ev_file.parent
        # Look for any eeg file with a similar base name
        eeg_files = list(eeg_dir.glob(f"{ev_name}*eeg.*"))
        if not eeg_files:
            eeg_files = list(eeg_dir.glob("*eeg.*")) # Fallback
            
        eeg_files = [f for f in eeg_files if f.suffix.lower() not in ['.json', '.tsv', '.csv', '.txt']]
        eeg_path = str(eeg_files[0].resolve()) if eeg_files else ""
        
        # 4. Parse the events
        try:
            df_ev = pd.read_csv(ev_file, sep='\t')
            for idx, row in df_ev.iterrows():
                # We save all row properties into a JSON metadata string so wrapper 
                # scripts can extract dataset-specific labels later
                row_dict = row.to_dict()
                
                # Standard fields
                onset = row.get("onset", 0.0)
                duration = row.get("duration", 0.0)
                trial_type = str(row.get("trial_type", ""))
                
                trials.append({
                    "trial_id": f"{session_uid}_run-{run}_{idx}",
                    "subject_id": sub_id,
                    "session_id": session_uid,
                    "dataset": dataset_name,
                    "task": task,
                    "eeg_path": eeg_path,
                    "onset": onset,
                    "duration": duration,
                    "trial_type": trial_type,
                    "raw_metadata": str(row_dict),  # for later processing
                    "split": "train" # Default, can be overridden by dataset wrapper
                })
        except Exception as e:
            logger.error(f"Error parsing events {ev_file}: {e}")

    # Deduplicate
    subjects_df = pd.DataFrame(subjects).drop_duplicates(subset=["subject_id"]) if subjects else pd.DataFrame()
    sessions_df = pd.DataFrame(sessions).drop_duplicates(subset=["session_id"]) if sessions else pd.DataFrame()
    trials_df = pd.DataFrame(trials) if trials else pd.DataFrame()
    
    return subjects_df, sessions_df, trials_df
