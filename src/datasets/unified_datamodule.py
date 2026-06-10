import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Optional, List, Tuple

class UnifiedEEGDataset(Dataset):
    def __init__(self, manifest_df: pd.DataFrame, data_dir: str):
        """
        manifest_df: DataFrame containing the rows from trials.csv
        data_dir: Path to data/processed/common_250hz
        """
        self.df = manifest_df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dataset = row["dataset"]
        trial_id = row["trial_id"]
        
        # We need to construct the path to the cached tensor
        tensor_path = self.data_dir / dataset / f"{trial_id}.pt"
        if not tensor_path.exists():
            raise FileNotFoundError(f"Missing cached tensor: {tensor_path}")
            
        x = torch.load(tensor_path, weights_only=True)
        
        # Simple dummy label for now (since downstream tasks may vary)
        label = 0
        
        return x, label

def collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pads a batch of EEG tensors to the maximum length in the batch.
    Returns:
        batch_x: (B, C, T_max)
        attention_mask: (B, T_max) boolean mask where True indicates REAL data, False indicates PADDING
        labels: (B,)
    """
    xs, labels = zip(*batch)
    
    # xs is a list of tensors of shape (C, T_i)
    lengths = [x.shape[1] for x in xs]
    max_len = max(lengths)
    
    C = xs[0].shape[0]
    B = len(xs)
    
    batch_x = torch.zeros(B, C, max_len, dtype=torch.float32)
    # Mask is True for actual data, False for padding
    attention_mask = torch.zeros(B, max_len, dtype=torch.bool)
    
    for i, x in enumerate(xs):
        T = lengths[i]
        batch_x[i, :, :T] = x
        attention_mask[i, :T] = True
        
    return batch_x, attention_mask, torch.tensor(labels, dtype=torch.long)

class UnifiedDataModule(pl.LightningDataModule):
    def __init__(self, manifest_path: str, data_dir: str, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage: Optional[str] = None):
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
            
        df = pd.read_csv(self.manifest_path)
        
        # Filter out rows that don't have cached files (e.g. ZuCo if it was skipped)
        # This prevents __getitem__ from crashing on missing files
        def check_exists(row):
            return (self.data_dir / row["dataset"] / f"{row['trial_id']}.pt").exists()
            
        # In a real massive dataset, we might want to pre-validate this faster, 
        # but for 5000 rows it's acceptable.
        valid_mask = df.apply(check_exists, axis=1)
        df = df[valid_mask]
        
        if len(df) == 0:
            raise ValueError(f"No valid cached tensors found in {self.data_dir}")
        
        # Split into 80/10/10
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        if stage == 'fit' or stage is None:
            self.train_dataset = UnifiedEEGDataset(train_df, self.data_dir)
            self.val_dataset = UnifiedEEGDataset(val_df, self.data_dir)
            
        if stage == 'test' or stage is None:
            self.test_dataset = UnifiedEEGDataset(test_df, self.data_dir)
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, 
                          num_workers=self.num_workers, collate_fn=collate_fn)
                          
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers, collate_fn=collate_fn)
                          
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers, collate_fn=collate_fn)
