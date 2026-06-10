# Experiment Plan

## Overview

Nine experiments (E1–E9) progressively validate the framework from baselines
through to deployment.

## Experiment Table

| ID | Name | Dataset(s) | Protocol | Model | Metric |
|----|------|-----------|----------|-------|--------|
| E1 | Single-dataset baselines | Each separately | Within-subj 5-fold | EEGNet, DeepConvNet, CNN-LSTM | Balanced Accuracy |
| E2 | Proposed classification | Each separately | Within-subj 5-fold | Conformer variants | Balanced Accuracy, macro F1 |
| E3 | Cross-subject generalization | Each separately | LOSO | Best from E2 | Balanced Accuracy |
| E4 | Pretrain → fine-tune | ZuCo → {KaraOne, ToL, Chisco} | LOSO | Pretrained Conformer | ΔAccuracy vs no pretrain |
| E5 | Multi-task & retrieval | KaraOne, Chisco | Within-subj | Full model | R@1, R@5, MRR |
| E6 | Sentence generation + RL | Chisco | Within-subj | Full + decoder + RL | WER, CER, Semantic Sim |
| E7 | Ablation study | KaraOne | LOSO | Ablation variants | Balanced Accuracy |
| E8 | Deployment profiling | KaraOne | — | Small, Medium, Full | Latency, Memory, RTF |
| E9 | Cross-dataset transfer | Train: {A, B} → Test: C | Cross-dataset | Pretrained encoder | Balanced Accuracy |

## Ablation Study (E7)

| ID | Ablation | Variants |
|----|----------|----------|
| A1 | Raw EEG vs CWT vs FilterBank | 3 variants |
| A2 | Without graph vs With graph | 2 variants |
| A3 | Transformer vs Conformer | 2 variants |
| A4 | Conformer vs Conformer+Mamba | 2 variants |
| A5 | Without CTC vs With CTC | 2 variants |
| A6 | Without pretraining vs With | 2 variants |
| A7 | No RL vs SCST vs PPO | 3 variants |
| A8 | Common-channel vs Region-pool vs Graph | 3 variants |
| A9 | Within-subject vs Cross-subject | 2 variants |

## Statistical Reporting

All results must include:
- Mean ± std across folds/subjects
- 95% confidence intervals (bootstrap, 10k resamples)
- McNemar's test (classification) or paired permutation test
- Bonferroni correction for multiple comparisons
- Effect size (Cohen's d)
