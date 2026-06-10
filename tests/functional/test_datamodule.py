import pytest
import torch
import pandas as pd
from pathlib import Path
from src.datasets.unified_datamodule import UnifiedEEGDataset, collate_fn, UnifiedDataModule

@pytest.fixture
def mock_cached_data(tmp_path):
    # Create a mock manifest
    manifest_dir = tmp_path / "data" / "processed" / "manifests"
    manifest_dir.mkdir(parents=True)
    
    trials = [
        {"dataset": "test_ds", "trial_id": "trial_1"},
        {"dataset": "test_ds", "trial_id": "trial_2"},
        {"dataset": "test_ds", "trial_id": "trial_3"},
    ]
    df = pd.DataFrame(trials)
    manifest_path = manifest_dir / "trials.csv"
    df.to_csv(manifest_path, index=False)
    
    # Create mock tensors
    data_dir = tmp_path / "data" / "processed" / "common_250hz"
    ds_dir = data_dir / "test_ds"
    ds_dir.mkdir(parents=True)
    
    # trial_1: length 100
    torch.save(torch.randn(61, 100), ds_dir / "trial_1.pt")
    # trial_2: length 250
    torch.save(torch.randn(61, 250), ds_dir / "trial_2.pt")
    # trial_3: length 150
    torch.save(torch.randn(61, 150), ds_dir / "trial_3.pt")
    
    return manifest_path, data_dir

class TestUnifiedDataModule:
    def test_dataset_loading(self, mock_cached_data):
        manifest_path, data_dir = mock_cached_data
        df = pd.read_csv(manifest_path)
        
        dataset = UnifiedEEGDataset(df, str(data_dir))
        assert len(dataset) == 3
        
        x, label = dataset[0]
        assert x.shape == (61, 100)
        assert label == 0
        
    def test_collate_fn(self):
        # Create dummy batch
        batch = [
            (torch.randn(61, 100), 0),
            (torch.randn(61, 250), 1),
            (torch.randn(61, 150), 0)
        ]
        
        batch_x, attention_mask, labels = collate_fn(batch)
        
        # Batch size 3, Channels 61, Max Time 250
        assert batch_x.shape == (3, 61, 250)
        assert attention_mask.shape == (3, 250)
        assert labels.shape == (3,)
        
        # Check masks
        # Trial 1
        assert attention_mask[0, :100].all()
        assert not attention_mask[0, 100:].any()
        
        # Trial 2
        assert attention_mask[1, :].all()
        
        # Trial 3
        assert attention_mask[2, :150].all()
        assert not attention_mask[2, 150:].any()
        
        # Check values are padded with 0
        assert (batch_x[0, :, 100:] == 0).all()

    def test_datamodule_setup(self, mock_cached_data):
        manifest_path, data_dir = mock_cached_data
        
        dm = UnifiedDataModule(str(manifest_path), str(data_dir), batch_size=2)
        dm.setup()
        
        # 3 total samples
        # 80/10/10 split will usually yield: Train 2, Val 0, Test 1 or similar for tiny datasets
        assert len(dm.train_dataset) > 0
        
        train_loader = dm.train_dataloader()
        batch_x, attention_mask, labels = next(iter(train_loader))
        
        assert batch_x.dim() == 3
        assert batch_x.size(0) <= 2  # batch size
        assert batch_x.size(1) == 61
