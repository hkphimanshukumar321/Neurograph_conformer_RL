import pandas as pd
import numpy as np
import torch
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(".").resolve()))
from src.datasets.manifest import ManifestDataset

def main():
    manifest_path = Path("data/processed/manifests/trials.csv")
    ds = ManifestDataset(manifest_path, dataset_name="Multiclass_Full_Run", split="train")
    
    print(f"Total samples: {len(ds)}")
    print(f"Number of classes: {ds.n_classes}")
    print(f"Label column used: {ds._label_col}")
    
    # Check what __getitem__ returns
    labels = []
    datasets = []
    for i in range(min(5000, len(ds))):
        sample = ds[i]
        labels.append(sample["label"].item())
        datasets.append(sample["dataset"])
        
    counts = Counter(labels)
    print("\nClass counts in first 5000 __getitem__ calls:")
    for k, v in counts.most_common(10):
        print(f"  Class {k} ({ds.label_int_to_str.get(k, '?')}): {v}")
        
    # Check weights
    weights = ds.get_class_weights()
    print(f"\nWeights min: {weights.min().item()}, max: {weights.max().item()}")
    print(f"Weights[:10]: {weights[:10].tolist()}")
    
if __name__ == "__main__":
    main()
