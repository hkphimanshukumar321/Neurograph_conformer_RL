import pandas as pd
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

def parse_chisco(root_path: str):
    """
    Custom wrapper for parsing the Chisco (ds005170) dataset.
    Since there are no events.tsv, each run-xxx_eeg.edf file is treated as a single trial.
    The labels are theoretically mapped from json/textmaps.json.
    """
    root_path = Path(root_path)
    if not root_path.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    subjects = []
    sessions = []
    trials = []
    
    textmap = {}
    textmap_file = root_path / "json" / "textmaps.json"
    if textmap_file.exists():
        try:
            with open(textmap_file, 'r', encoding='utf-8') as f:
                # Assuming JSON is a dictionary mapping run string -> text
                textmap = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load textmaps.json: {e}")
            
    # Find all subjects
    for sub_dir in root_path.glob("sub-*"):
        if not sub_dir.is_dir(): continue
        sub_id = sub_dir.name
        
        subjects.append({
            "subject_id": sub_id,
            "dataset": "chisco",
            "age": "n/a",
            "sex": "n/a"
        })
        
        # Find all sessions
        for ses_dir in sub_dir.glob("ses-*"):
            if not ses_dir.is_dir(): continue
            ses_id = ses_dir.name
            session_uid = f"chisco_{sub_id}_{ses_id}"
            
            sessions.append({
                "session_id": session_uid,
                "subject_id": sub_id,
                "dataset": "chisco",
                "task": "imagine" 
            })
            
            eeg_dir = ses_dir / "eeg"
            if not eeg_dir.exists(): continue
            
            # Find all .edf files
            for edf_file in eeg_dir.glob("*_eeg.edf"):
                # e.g. sub-02_ses-02_task-imagine_run-011_eeg.edf
                name_parts = edf_file.name.replace("_eeg.edf", "").split("_")
                run = "unknown"
                for p in name_parts:
                    if p.startswith("run-"):
                        run = p.split("-")[1]
                        
                trial_id = f"{session_uid}_run-{run}"
                
                # Attempt to get a label from textmap if possible
                try:
                    run_int = int(run)
                    label = textmap.get(str(run_int), str(run))
                except ValueError:
                    label = textmap.get(run, run)
                    
                trials.append({
                    "trial_id": trial_id,
                    "subject_id": sub_id,
                    "session_id": session_uid,
                    "dataset": "chisco",
                    "task": "imagine",
                    "eeg_path": str(edf_file.resolve()),
                    "onset": 0.0,
                    "duration": -1.0, # Indicates whole file
                    "trial_type": label,
                    "raw_metadata": "{}",
                    "split": "train"
                })
                
    subjects_df = pd.DataFrame(subjects).drop_duplicates(subset=["subject_id"]) if subjects else pd.DataFrame()
    sessions_df = pd.DataFrame(sessions).drop_duplicates(subset=["session_id"]) if sessions else pd.DataFrame()
    trials_df = pd.DataFrame(trials).drop_duplicates(subset=["trial_id"]) if trials else pd.DataFrame()
    
    return subjects_df, sessions_df, trials_df
