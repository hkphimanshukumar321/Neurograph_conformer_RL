# Datasets

## Overview

Four EEG datasets spanning three languages and complementary task paradigms.

> **Critical**: ZuCo is NOT an imagined speech dataset. It captures natural
> reading EEG and is used ONLY for self-supervised pretraining (Stage 1).

## Dataset Summary

| Dataset | Language | Task | N_subj | Channels | Sfreq | Classes | Paradigm |
|---------|----------|------|--------|----------|-------|---------|----------|
| Kara One | English | Imagined speech | 14 | 62 (10-20) | 500 Hz | 7 phonemes + 4 words = 11 | Lab EEG |
| Thinking Out Loud | English | Inner speech | 10 | 14 (EPOC) | 128 Hz | 4 directional words | Consumer EEG |
| Chisco | Chinese | Imagined speech sentences | 20 | 64 (10-20) | 1000 Hz | ~80 characters | Lab EEG |
| ZuCo | English | Natural reading | 30 | 128 (GSN) | 500 Hz | N/A (pretraining only) | Lab EEG |

## Dataset Details

### Kara One (Zhao & Rudzicz, 2015)
- **URL**: http://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html
- **Task**: Imagined speech of 7 phonemes (/iy/, /uw/, /piy/, /tiy/, /diy/, /m/, /n/) and 4 words ("pat", "pot", "knew", "gnaw")
- **Protocol**: Within-subject 5-fold CV + LOSO
- **Key challenge**: Small dataset (~100 trials/subject), phoneme-level classification

### Thinking Out Loud (Nieto et al., 2022)
- **URL**: https://openneuro.org/datasets/ds003626
- **Task**: Inner speech of 4 directional words ("up", "down", "left", "right")
- **Protocol**: Within-subject 5-fold CV + LOSO
- **Key challenge**: Consumer-grade EEG (14 channels, limited spatial resolution)

### Chisco (Chinese Inner Speech Corpus)
- **Task**: Imagined speech of Chinese sentences
- **Protocol**: Sentence-level decoding with sliding window
- **Key challenge**: Cross-lingual, character-level output, variable-length trials

### ZuCo (Hollenstein et al., 2018, 2020)
- **URL**: https://osf.io/q3zws/
- **Task**: Natural reading with eye-tracking co-registration
- **Used for**: Self-supervised pretraining ONLY (EEG-text contrastive alignment)
- **Why not imagined speech?**: ZuCo captures passive reading, not active speech imagery

## Preprocessing Harmonization

All datasets are standardized to:
- Sampling rate: 250 Hz
- Bandpass: 0.5–100 Hz
- Reference: Common Average Reference (CAR)
- Normalization: Per-session z-score

Channel handling:
- Common intersection: 9 shared electrodes (loses information)
- Region pooling: 9 canonical brain regions (preferred)
- Graph with variable nodes: Full channel sets with position-based adjacency
