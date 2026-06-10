import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def parse_zuco(root_path: str):
    """
    Custom wrapper for parsing the ZuCo dataset.
    ZuCo uses large .mat files containing nested struct arrays of sentences.
    This parser registers 1 session/trial per .mat file. 
    The PyTorch ZuCoDataset class will handle unpacking the sentences later.
    """
    root_path = Path(root_path)
    if not root_path.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    subjects = []
    sessions = []
    trials = []
    
    mat_files = list(root_path.rglob("*.mat"))
    
    for mat_file in mat_files:
        name_lower = mat_file.name.lower()
        if "task1" not in name_lower and "task2" not in name_lower and "task3" not in name_lower and "nr" not in name_lower and "tsr" not in name_lower:
            continue
            
        # Try to guess task and subject from filename. e.g. task1-SR-ZAB.mat or resultsZAB_NR.mat
        task = "unknown"
        if "task1" in name_lower or "sr" in name_lower: task = "SR"
        if "task2" in name_lower or "nr" in name_lower: task = "NR"
        if "task3" in name_lower or "tsr" in name_lower: task = "TSR"
        
        # ZuCo subject IDs are typically 3 uppercase letters starting with Z (e.g. ZAB, ZPH)
        sub_id = "unknown"
        for part in mat_file.stem.split("_"):
            for p2 in part.split("-"):
                if len(p2) == 3 and p2.startswith("Z") and p2.isupper():
                    sub_id = p2
                    
        # If we couldn't find a subject ID, try looking at the parent folder
        if sub_id == "unknown" and len(mat_file.parent.name) == 3 and mat_file.parent.name.startswith("Z"):
            sub_id = mat_file.parent.name
            
        session_uid = f"zuco_{sub_id}_{task}"
        
        subjects.append({
            "subject_id": sub_id,
            "dataset": "zuco",
            "age": "n/a",
            "sex": "n/a"
        })
        
        sessions.append({
            "session_id": session_uid,
            "subject_id": sub_id,
            "dataset": "zuco",
            "task": task
        })
        
        trials.append({
            "trial_id": f"{session_uid}_full",
            "subject_id": sub_id,
            "session_id": session_uid,
            "dataset": "zuco",
            "task": task,
            "eeg_path": str(mat_file.resolve()),
            "onset": 0,
            "duration": -1, # Indicates whole file
            "trial_type": "all_sentences",
            "raw_metadata": "{}",
            "split": "train"
        })
        
    subjects_df = pd.DataFrame(subjects).drop_duplicates(subset=["subject_id"]) if subjects else pd.DataFrame()
    sessions_df = pd.DataFrame(sessions).drop_duplicates(subset=["session_id"]) if sessions else pd.DataFrame()
    trials_df = pd.DataFrame(trials).drop_duplicates(subset=["trial_id"]) if trials else pd.DataFrame()
    
    return subjects_df, sessions_df, trials_df
