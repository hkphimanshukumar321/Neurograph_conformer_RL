# NeuroGraph-Conformer-RL

> **Multi-Level Neural Language Decoding from Non-Invasive EEG via
> Graph-Conformer Architecture with Sequence-Level Reward Optimization**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

NeuroGraph-Conformer is a multi-level framework for decoding imagined/inner
speech from scalp EEG. Unlike single-task EEG classifiers, this project
addresses the full decoding pipeline:

```
Raw EEG → Preprocessing → Time-Frequency → Spatial Graph → Conformer+Mamba
        → Latent Brain-Speech State → Classification / Retrieval / Generation
        → RL-Optimized Text Output
```

### Key Contributions

1. **Graph-Conformer encoder** combining GATv2 spatial modeling with
   Conformer temporal modeling and optional Mamba long-range refinement
2. **Multi-task framework** supporting classification (T1), retrieval (T2),
   and generation (T3) from a shared encoder
3. **Cross-dataset harmonization** via region pooling (9 canonical brain
   regions) to handle heterogeneous electrode montages
4. **Sequence-level reward optimization** using SCST/MWER/PPO for
   generation quality beyond cross-entropy
5. **Deployment-ready export** with ONNX/INT8 quantization and streaming
   inference profiling

## Architecture

| Component | Description |
|-----------|-------------|
| **Frontend** | CWT (Morlet) or learnable SincNet filter bank |
| **Spatial** | GATv2 graph network with 3D position-based adjacency |
| **Temporal** | Conformer (Macaron FFN + MHSA + DepthwiseConv) |
| **Sequential** | Mamba SSM for O(L) long-range dependencies |
| **Decoder** | Transformer decoder with CTC auxiliary branch |
| **RL** | SCST with multi-reward: semantic similarity + WER + CER |

### Model Variants

| Variant | Params | Tasks | Use Case |
|---------|--------|-------|----------|
| Small | ~0.5M | Classification | Deployment / real-time |
| Medium | ~5M | Classification + Retrieval | Research |
| Full | ~25M | All (Classification + Retrieval + Generation) | Full paper |

## Datasets

| Dataset | Lang | Channels | Classes | Role |
|---------|------|----------|---------|------|
| [Thinking Out Loud](https://openneuro.org/datasets/ds003626) | EN | 10 | Commands/Imagined Speech | Primary benchmark |
| [Thinking Out Loud](https://openneuro.org/datasets/ds003626) | EN | 14 (EPOC) | 4 directional words | Consumer EEG test |
| Chisco | ZH | 64 | ~80 characters | Sentence generation |
| [ZuCo](https://osf.io/q3zws/) | EN | 128 | N/A | **Pretraining only** |

> ⚠️ **ZuCo is NOT an imagined speech dataset.** It captures natural reading
> and is used exclusively for self-supervised pretraining.

## Installation

```bash
# Clone
git clone https://github.com/yourname/NeuroGraph-Conformer-RL.git
cd NeuroGraph-Conformer-RL

# Create environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install project in development mode
pip install -e .
```

## Project Structure

```
NeuroGraph-Conformer-RL/
├── configs/                    # Hierarchical YAML configurations
│   ├── base.yaml               # Global defaults
│   ├── datasets/               # Dataset-specific configs
│   ├── models/                 # Model architecture configs
│   ├── training/               # Training stage configs
│   └── deployment/             # Export/quantization configs
├── src/                        # Source code
│   ├── preprocessing/          # EEG preprocessing pipeline
│   ├── features/               # Feature extraction (region pooling)
│   ├── datasets/               # Dataset loaders
│   ├── models/                 # Model architectures
│   │   ├── neurograph.py       # Main model assembly
│   │   ├── frontend.py         # CWT / FilterBank front-ends
│   │   ├── graph_encoder.py    # GATv2 spatial encoder
│   │   ├── conformer.py        # Conformer temporal encoder
│   │   ├── mamba_module.py     # Mamba SSM module
│   │   ├── transformer_decoder.py  # Autoregressive decoder
│   │   ├── heads.py            # Output heads (cls/retrieval/gen)
│   │   ├── losses.py           # All loss functions
│   │   └── baselines/          # 7 baseline models
│   ├── training/               # Training loop & RL
│   ├── evaluation/             # Metrics, statistics, visualization
│   └── deployment/             # Export, quantization, streaming
├── scripts/                    # CLI entry points
│   ├── train.py                # Training
│   ├── preprocess.py           # Preprocessing
│   ├── evaluate.py             # Evaluation
│   ├── run_ablation.py         # Ablation studies
│   ├── export_model.py         # Model export
│   └── profile_model.py        # Inference profiling
├── tests/                      # Unit tests
├── docs/                       # Documentation
├── notebooks/                  # Jupyter notebooks
└── data/                       # Data directory (not in git)
```

## Usage

### 1. Link Raw Data
Ensure your downloaded data is on the server, then run the scan script:
```bash
bash scripts/01_scan_datasets.sh
```

### 2. Prepare Metadata
Initialize the standard metadata structures:
```bash
bash scripts/02_prepare_metadata.sh
```

### 3. Preprocess and Cache
Run the preprocessing pipeline on the linked data to produce `.pt` caches:
```bash
python scripts/03_preprocess_and_cache.py --project_root .
```

### 4. Train Classification Model
Start the training loop in the background:
```bash
bash scripts/04_train.sh chisco First_Run
```

### 5. Run Ablation Study

```bash
python scripts/run_ablation.py \
    --config configs/training/train_cls.yaml \
    --dataset kara_one \
    --ablations A1,A2,A3,A4
```

### 4. Export for Deployment

```bash
python scripts/export_model.py \
    --checkpoint results/checkpoints/best.pt \
    --format onnx \
    --quantize int8
```

## Training Stages

| Stage | Script | Config | Description |
|-------|--------|--------|-------------|
| 1 | `train.py` | `pretrain_ssl.yaml` | Self-supervised EEG-text contrastive pretraining on ZuCo |
| 2 | `train.py` | `train_cls.yaml` | Supervised classification per dataset |
| 3 | `train.py` | `train_seq.yaml` | Sequence training (CTC + CE) for generation |
| 4 | `train.py` | `train_rl.yaml` | RL fine-tuning with SCST/MWER/PPO |
| 5 | `train.py` | `finetune_xsubject.yaml` | Cross-subject few-shot adaptation |

## Baselines

7 baseline models for fair comparison:

| Model | Type | Reference |
|-------|------|-----------|
| EEGNet | Compact CNN | Lawhern et al., 2018 |
| DeepConvNet | Deep CNN | Schirrmeister et al., 2017 |
| CNN-LSTM | CNN + LSTM | — |
| CNN-BiGRU | CNN + BiGRU | — |
| Vanilla Transformer | Attention only | Vaswani et al., 2017 |
| Graph-only | GATv2 + MLP | — |
| Mamba-only | SSM only | Gu & Dao, 2023 |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_models.py -v
```

## Citation

```bibtex
@article{neurograph2025,
  title={Multi-Level Neural Language Decoding from Non-Invasive EEG via
         Graph-Conformer Architecture with Sequence-Level Reward Optimization},
  author={Your Name},
  year={2025},
  journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Kara One dataset: Zhao & Rudzicz (2015)
- ZuCo dataset: Hollenstein et al. (2018, 2020)
- Thinking Out Loud: Nieto et al. (2022)
- Conformer: Gulati et al. (2020)
- Mamba: Gu & Dao (2023)
