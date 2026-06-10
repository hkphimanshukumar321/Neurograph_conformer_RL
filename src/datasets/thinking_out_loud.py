import pandas as pd
import ast
from src.datasets.bids_loader import parse_bids_dataset

def parse_thinking_out_loud(root_path: str):
    """
    Wrapper for parsing the Thinking Out Loud (ds003626) BIDS dataset.
    Extracts inner speech labels and conditions.
    """
    subjects, sessions, trials = parse_bids_dataset("thinking_out_loud", root_path)
    
    if trials.empty:
        return subjects, sessions, trials
        
    def extract_label(row):
        try:
            meta = ast.literal_eval(row['raw_metadata'])
            # ds003626 typically uses 'trial_type' or 'stim_file' to denote the phoneme/word
            return meta.get('trial_type', meta.get('value', ''))
        except Exception:
            return row['trial_type']
            
    trials['label'] = trials.apply(extract_label, axis=1)
    return subjects, sessions, trials
