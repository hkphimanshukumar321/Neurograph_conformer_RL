import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def _load_events_dat(event_file: Path) -> np.ndarray:
    """
    Load an MNE events.dat file. These are numpy binary arrays
    with shape (n_events, 3): [sample_idx, prev_value, event_id].
    """
    try:
        # MNE typically saves events with np.save()
        events = np.load(str(event_file), allow_pickle=False)
        if events.ndim == 2 and events.shape[1] >= 3:
            return events
        elif events.ndim == 1:
            # Reshape if flat
            events = events.reshape(-1, 3)
            return events
    except Exception:
        pass
    
    try:
        # Fallback: some older MNE versions use np.save with allow_pickle
        events = np.load(str(event_file), allow_pickle=True)
        if hasattr(events, 'shape'):
            if events.ndim == 2 and events.shape[1] >= 3:
                return events
            elif events.ndim == 0:
                # Pickled object — try to extract
                obj = events.item()
                if isinstance(obj, np.ndarray):
                    return obj.reshape(-1, 3)
    except Exception:
        pass
    
    try:
        # Last fallback: raw binary int32
        raw = np.fromfile(str(event_file), dtype=np.int32)
        if len(raw) % 3 == 0 and len(raw) > 0:
            return raw.reshape(-1, 3)
    except Exception:
        pass
    
    return np.array([]).reshape(0, 3)


def parse_thinking_out_loud(root_path: str):
    """
    Custom wrapper for parsing Thinking Out Loud (ds003626).
    Events are stored as binary numpy .dat files in the derivatives/ folder.
    """
    root_path = Path(root_path)
    if not root_path.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    subjects = []
    sessions = []
    trials = []
    
    # Scan raw subjects first
    for sub_dir in sorted(root_path.glob("sub-*")):
        if not sub_dir.is_dir(): continue
        sub_id = sub_dir.name
        
        subjects.append({
            "subject_id": sub_id,
            "dataset": "thinking_out_loud",
            "age": "n/a",
            "sex": "n/a"
        })
        
        for ses_dir in sorted(sub_dir.glob("ses-*")):
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
                events = _load_events_dat(event_file)
                if events.shape[0] > 0:
                    logger.info(f"  Loaded {events.shape[0]} events from {event_file.name}")
                    for idx in range(events.shape[0]):
                        sample_idx = int(events[idx, 0])
                        event_id = int(events[idx, 2]) if events.shape[1] > 2 else int(events[idx, 1])
                        
                        trials.append({
                            "trial_id": f"{session_uid}_ev-{idx}",
                            "subject_id": sub_id,
                            "session_id": session_uid,
                            "dataset": "thinking_out_loud",
                            "task": "innerspeech",
                            "eeg_path": eeg_path,
                            "onset": sample_idx,
                            "duration": 0,
                            "trial_type": str(event_id),
                            "raw_metadata": f'{{"event_file": "{str(event_file.resolve())}"}}',
                            "split": "train"
                        })
                else:
                    logger.warning(f"  Could not decode events from {event_file}")
                    # Fallback: register the whole session as one trial
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
            else:
                # No events file at all — register whole session
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
