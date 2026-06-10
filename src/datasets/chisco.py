import pandas as pd
import ast
from src.datasets.bids_loader import parse_bids_dataset

def parse_chisco(root_path: str):
    """
    Wrapper for parsing the Chisco (ds005170) BIDS dataset.
    Extracts relevant Chinese imagined speech semantic labels.
    """
    subjects, sessions, trials = parse_bids_dataset("chisco", root_path)
    
    if trials.empty:
        return subjects, sessions, trials
        
    def extract_label(row):
        try:
            meta = ast.literal_eval(row['raw_metadata'])
            # Chisco might store the label in 'trial_type', 'value', or a custom column
            return meta.get('trial_type', meta.get('value', ''))
        except Exception:
            return row['trial_type']
            
    trials['label'] = trials.apply(extract_label, axis=1)
    return subjects, sessions, trials
