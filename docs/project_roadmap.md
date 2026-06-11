# Project Roadmap & Implementation Order

## 1. Implementation Order

Implement in this order:
1. `scripts/00_make_project_tree.sh`
2. `scripts/01_scan_datasets.sh`
3. `scripts/02_prepare_metadata.sh`
4. `scripts/03_preprocess_and_cache.py` (Multiprocessing CPU)
5. `scripts/04_train.sh`
6. `scripts/05_run_all_end_to_end.sh` (Master)
7. `src/preprocessing/io_loaders.py`
6. `src/preprocessing/filters.py`
7. `src/preprocessing/channel_harmonization.py`
8. `src/features/wavelet.py`
9. `src/features/graph_builder.py`
10. `src/datasets/karaone.py`
11. `src/datasets/zuco.py`
12. `src/datasets/unified_datamodule.py`
13. `src/models/eegnet.py`
14. `src/models/transformer_baseline.py`
15. `src/models/graph_encoder.py`
16. `src/models/conformer_encoder.py`
17. `src/models/wavegraph_conformer.py`
18. `src/evaluation/metrics_classification.py`
19. `src/evaluation/metrics_text.py`
20. `src/evaluation/metrics_retrieval.py`

**Staged Approach:**
Do not start with PPO. First prove:
`EEGNet < Transformer < Conformer < Wavelet+Graph+Conformer`

Then add:
- Mamba
- CTC
- Retrieval loss
- SCST/MWER
- PPO only at the final stage

---

## 2. IEEE-Reviewer Experimental Checklist

Your experiments must include these, otherwise the work will look over-engineered:

| Experiment | Required? | Why |
| :--- | :--- | :--- |
| **Within-dataset baseline** | Yes | Shows your method works before merging |
| **Cross-subject split** | Yes | Prevents inflated subject-dependent accuracy |
| **Provider-preprocessing baseline** | Yes | Fair comparison with dataset-original methods |
| **Unified-preprocessing baseline** | Yes | Shows your common-ground pipeline is valid |
| **Wavelet ablation** | Yes | Proves time-frequency front-end matters |
| **Graph ablation** | Yes | Proves spatial modelling matters |
| **Conformer vs Transformer** | Yes | Proves architecture choice |
| **Mamba ablation** | Yes | Proves long-sequence module is useful |
| **RL ablation** | Yes, later | Proves sequence reward helps |
| **Deployment profiling** | Yes | Supports “workable/deployable” claim |

---

## 3. Final Metric Plan

**KARA ONE:**
- Accuracy, balanced accuracy, macro-F1, confusion matrix, top-k accuracy

**Thinking Out Loud:**
- Accuracy, balanced accuracy, macro-F1, top-k accuracy

**Chisco:**
- Semantic category accuracy, Recall@k, MRR, semantic similarity, CER/WER if generation is attempted

**ZuCo:**
- Reading-task classification, sentence retrieval, embedding alignment, semantic similarity

**Deployment:**
- Parameters, FLOPs, latency, memory, calibration time

---

## 4. Practical Command Sequence on Server

```bash
# 1. Create project
bash scripts/00_make_project_tree.sh /path/to/eeg-speech-decoding
cd /path/to/eeg-speech-decoding

# 2. End-to-End Execution
# This automatically handles scanning, metadata, parallel caching, and sequential training.
bash scripts/05_run_all_end_to_end.sh
```
