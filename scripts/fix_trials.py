import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def fix_trials_csv():
    project_root = Path(__file__).resolve().parent.parent
    trials_csv = project_root / "data" / "processed" / "manifests" / "trials.csv"
    
    if not trials_csv.exists():
        logger.error(f"Cannot find {trials_csv}")
        return
        
    df = pd.read_csv(trials_csv)
    initial_len = len(df)
    
    # Filter out anything that isn't a valid raw EEG format
    valid_extensions = ['.edf', '.bdf', '.vhdr', '.mat', '.set']
    
    def is_valid_eeg(path_str):
        if pd.isna(path_str): return False
        return Path(str(path_str)).suffix.lower() in valid_extensions
        
    source_col = "eeg_path" if "eeg_path" in df.columns else "source_file"
    df = df[df[source_col].apply(is_valid_eeg)]
    
    final_len = len(df)
    removed = initial_len - final_len
    
    df.to_csv(trials_csv, index=False)
    logger.info(f"Fixed trials.csv! Removed {removed} invalid entries. Remaining valid EEG trials: {final_len}")

if __name__ == "__main__":
    fix_trials_csv()
