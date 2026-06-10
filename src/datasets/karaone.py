import pandas as pd
import logging

logger = logging.getLogger(__name__)

def parse_karaone(root_path: str):
    """
    Stub wrapper for parsing the KARA ONE dataset.
    Requires mne and scipy for reading .cnt and epoch_inds.mat files.
    """
    logger.info(f"KARA ONE parsing not yet fully implemented for {root_path}")
    # Return empty dataframes for now
    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
