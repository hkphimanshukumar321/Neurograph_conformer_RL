# NeuroGraph-Conformer Architecture

## Overview

NeuroGraph-Conformer is a multi-level neural language decoding framework for
non-invasive EEG-based imagined/inner speech recognition.

## Pipeline

```
Raw EEG (C, T)
  │
  ├── Preprocessing: 0.5–100 Hz bandpass, CAR, 250 Hz resample, ICA
  │
  ▼
Frontend (Time-Frequency Decomposition)
  ├── CWT (Morlet wavelet) → (C, F, T')
  └── FilterBank (learnable SincNet) → (C, n_bands)
  │
  ▼
Spatial Encoder (GATv2 Graph Network)
  ├── Distance-based adjacency from 3D electrode positions
  ├── Correlation-based adjacency from EEG signals
  └── Hybrid α·A_dist + (1-α)·A_corr
  │   Output: (batch, C, d_model)
  │
  ▼
Temporal Encoder (Conformer)
  ├── Macaron FFN (½)
  ├── Multi-Head Self-Attention (relative positional encoding)
  ├── Convolution Module (depthwise + pointwise)
  └── Macaron FFN (½)
  │   Output: (batch, L, d_model)
  │
  ▼
Sequential Refinement (Mamba SSM, optional)
  ├── Selective State Space Model
  └── O(L) complexity for long sequences
  │   Output: (batch, L, d_model)
  │
  ▼
Output Heads
  ├── T1: Classification → phoneme/word logits
  ├── T2: Retrieval → semantic embedding (contrastive)
  └── T3: Generation → autoregressive text (CTC + CE)

Optional Stage 4:
  └── RL Fine-tuning (SCST/MWER/PPO)
      └── Multi-reward: semantic similarity + WER + CER + fluency
```

## Model Variants

| Variant | Frontend    | Spatial      | Encoder       | Mamba | Decoder | ~Params |
|---------|-------------|--------------|---------------|-------|---------|---------|
| Small   | FilterBank  | RegionPool   | 2-layer, d=64 | No    | No      | 0.5M    |
| Medium  | CWT         | GATv2        | 4-layer, d=128| 1-layer| No     | 5M      |
| Full    | CWT+FB      | GATv2 Hybrid | 6-layer, d=256| 2-layer| TransDec| 25M    |

## Channel Strategy

### Region Pooling (9 regions)

Maps variable electrode montages to 9 canonical brain regions:

| Region | Abbr | Key 10-20 Electrodes | Function |
|--------|------|---------------------|----------|
| Left Frontal | LF | Fp1, AF3, F3, F7, FC3 | Speech planning, Broca's area |
| Right Frontal | RF | Fp2, AF4, F4, F8, FC4 | Prosody |
| Left Central | LC | C1, C3, CP1, CP3 | Articulatory motor cortex |
| Midline Central | MC | Fz, Cz, FCz, CPz | Supplementary motor area |
| Right Central | RC | C2, C4, CP2, CP4 | Contralateral motor |
| Left Temporal | LT | T7, TP7, FT7 | Wernicke's area, auditory |
| Right Temporal | RT | T8, TP8, FT8 | Auditory cortex |
| Parietal | PA | P3, Pz, P4, POz | Sensory integration |
| Occipital | OC | O1, Oz, O2 | Visual cortex |
