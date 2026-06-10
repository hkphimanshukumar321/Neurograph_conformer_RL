import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def parse_thinking_out_loud(root_path: str):
    """
    Custom wrapper for parsing Thinking Out Loud (ds003626).
    Events are stored as .dat files in the derivatives/ folder instead of standard BIDS.
    """
    root_path = Path(root_path)
    if not root_path.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    subjects = []
    sessions = []
    trials = []
    
    # Scan raw subjects first
    for sub_dir in root_path.glob("sub-*"):
        if not sub_dir.is_dir(): continue
        sub_id = sub_dir.name
        
        subjects.append({
            "subject_id": sub_id,
            "dataset": "thinking_out_loud",
            "age": "n/a",
            "sex": "n/a"
        })
        
        for ses_dir in sub_dir.glob("ses-*"):
            if not ses_dir.is_dir(): continue
            ses_id = ses_dir.name
            session_uid = f"tol_{sub_id}_{ses_id}"
            
            sessions.append({
                "session_id": session_uid,
                "subject_id": sub_id,
                "dataset": "thinking_out_loud",
                "task": "innerspeech"
            })
            
            eeg_dir = ses_dir / "eeg"
            eeg_files = list(eeg_dir.glob("*_eeg.bdf")) + list(eeg_dir.glob("*_eeg.fif"))
            eeg_path = str(eeg_files[0].resolve()) if eeg_files else ""
            
            # Find the corresponding events.dat in derivatives/
            event_file = root_path / "derivatives" / sub_id / ses_id / f"{sub_id}_{ses_id}_events.dat"
            if event_file.exists():
                try:
                    # MNE events.dat is usually space-separated or tab-separated without header
                    df_ev = pd.read_csv(event_file, sep=r'\s+', header=None, engine='python')
                    for idx, row in df_ev.iterrows():
                        event_sample = row[0]
                        # MNE usually has 3 columns: sample_idx, previous_val, event_id
                        event_id = row[2] if len(row) > 2 else row[1]
                        
                        trials.append({
                            "trial_id": f"{session_uid}_ev-{idx}",
                            "subject_id": sub_id,
                            "session_id": session_uid,
                            "dataset": "thinking_out_loud",
                            "task": "innerspeech",
                            "eeg_path": eeg_path,
                            "onset": event_sample, # Saving sample index as onset
                            "duration": 0,
                            "trial_type": str(event_id),
                            "raw_metadata": f'{{"event_file": "{str(event_file.resolve())}"}}',
                            "split": "train"
                        })
                except Exception as e:
                    logger.warning(f"Failed to read events {event_file}: {e}")
            else:
                # Add one placeholder trial if no events exist
                trials.append({
                    "trial_id": f"{session_uid}_full",
                    "subject_id": sub_id,
                    "session_id": session_uid,
                    "dataset": "thinking_out_loud",
                    "task": "innerspeech",
                    "eeg_path": eeg_path,
                    "onset": 0,
                    "duration": -1,
                    "trial_type": "unknown",
                    "raw_metadata": "{}",
                    "split": "train"
                })
                
    subjects_df = pd.DataFrame(subjects).drop_duplicates(subset=["subject_id"]) if subjects else pd.DataFrame()
    sessions_df = pd.DataFrame(sessions).drop_duplicates(subset=["session_id"]) if sessions else pd.DataFrame()
    trials_df = pd.DataFrame(trials).drop_duplicates(subset=["trial_id"]) if trials else pd.DataFrame()
    
    return subjects_df, sessions_df, trials_df
