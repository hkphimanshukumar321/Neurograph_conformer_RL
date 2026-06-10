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
    
    # Look for all .mat files
    mat_files = list(root_path.rglob("*.mat"))
    
    if not mat_files:
        logger.warning(f"No .mat files found in {root_path}!")
        # Dump skeleton so we can see what IS there
        logger.info("--- ZuCo SKELETON DUMP ---")
        try:
            all_files = [f for f in root_path.rglob("*") if f.is_file()]
            for f in all_files[:50]:
                logger.info(f"  {f.relative_to(root_path)}")
            if len(all_files) > 50:
                logger.info(f"  ... and {len(all_files) - 50} more files.")
            elif len(all_files) == 0:
                logger.info("  (empty directory)")
        except Exception as e:
            logger.error(f"Failed to dump skeleton: {e}")
        logger.info("--------------------------")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    logger.info(f"  Found {len(mat_files)} .mat files")
    
    for mat_file in mat_files:
        name_lower = mat_file.name.lower()
        stem = mat_file.stem
        
        # --- Determine task ---
        task = "unknown"
        # ZuCo 1.0 naming: resultsXXX_task1.mat, resultsXXX_task2.mat, resultsXXX_task3.mat
        # ZuCo 2.0 naming: resultsXXX_NR.mat, resultsXXX_TSR.mat
        # Also possible: task1-SR-ZAB.mat
        if "task1" in name_lower or "_sr" in name_lower or "-sr" in name_lower:
            task = "SR"
        elif "task2" in name_lower:
            task = "task2"
        elif "task3" in name_lower:
            task = "task3"
        elif "nr" in name_lower:
            task = "NR"
        elif "tsr" in name_lower:
            task = "TSR"
        # If we can't determine the task, just accept the file with "unknown" task
        
        # --- Determine subject ID ---
        # ZuCo subject IDs: 3 uppercase letters starting with Z (ZAB, ZPH, ZDM, etc.)
        sub_id = "unknown"
        # Try splitting on common delimiters
        for part in stem.replace("-", "_").split("_"):
            if len(part) == 3 and part[0].isupper() and part.isupper():
                sub_id = part
                break
        
        # Fallback: try to find a 3-letter uppercase sequence anywhere in the stem
        if sub_id == "unknown":
            import re
            matches = re.findall(r'[A-Z]{3}', stem)
            if matches:
                sub_id = matches[0]
                
        # Fallback: parent folder name
        if sub_id == "unknown":
            parent = mat_file.parent.name
            if len(parent) >= 2 and parent[0].isupper():
                sub_id = parent
                
        # Last resort: use filename as subject
        if sub_id == "unknown":
            sub_id = stem[:10]
        
        session_uid = f"zuco_{sub_id}_{task}"
        trial_id = f"zuco_{sub_id}_{task}_{mat_file.name}"
        
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
            "trial_id": trial_id,
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
