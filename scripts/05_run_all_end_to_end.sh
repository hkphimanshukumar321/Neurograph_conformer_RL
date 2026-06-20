#!/bin/bash
# ==============================================================================
# 05_run_all_end_to_end.sh — Master script for the full NeuroGraph-Conformer pipeline.
#
# This script executes:
#   0. Dataset download (Thinking Out Loud, Chisco, ZuCo)
#   1. Dataset scanning
#   2. Metadata extraction (build_manifests.py + fix_trials.py)
#   3. Parallel Preprocessing & Caching (CPU-bound)
#   4. Sequential Training (GPU-bound, sequential to avoid OOM)
#
# Usage:
#   nohup bash scripts/05_run_all_end_to_end.sh [experiment_name] > pipeline.log 2>&1 &
#
# Options:
#   --skip-download    Skip the download step (if data is already present)
#   --skip-preprocess  Skip the preprocessing step (if .pt caches already exist)
# ==============================================================================
set -e  # Exit immediately if a command exits with a non-zero status

# ── Resolve paths from script location (not CWD) ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"
export PROJECT_ROOT

EXPERIMENT_NAME="${1:-End_to_End_Run}"
SKIP_DOWNLOAD=0
SKIP_PREPROCESS=0

for arg in "$@"; do
  case "$arg" in
    --skip-download)   SKIP_DOWNLOAD=1 ;;
    --skip-preprocess) SKIP_PREPROCESS=1 ;;
  esac
done

echo "==========================================================="
echo "   NeuroGraph-Conformer End-to-End Pipeline Initialization "
echo "==========================================================="
echo "[INFO] PROJECT_ROOT  = $PROJECT_ROOT"
echo "[INFO] EXPERIMENT    = $EXPERIMENT_NAME"
echo "[INFO] SKIP_DOWNLOAD = $SKIP_DOWNLOAD"
echo ""

# ──────────────────────────────────────────────────────────────────
# STEP 0: Download all 3 datasets
# ──────────────────────────────────────────────────────────────────
if [[ "$SKIP_DOWNLOAD" == "0" ]]; then
  echo "[STEP 0/4] Downloading Datasets..."
  bash "$SCRIPT_DIR/00_download_all_datasets.sh"
  echo ""
else
  echo "[STEP 0/4] Skipping dataset download (--skip-download)"
  echo ""
fi

# ──────────────────────────────────────────────────────────────────
# STEP 1: Scan for datasets on disk
# ──────────────────────────────────────────────────────────────────
echo "[STEP 1/4] Scanning Server Datasets..."
export SEARCH_ROOTS="$PROJECT_ROOT/data/raw:$PROJECT_ROOT/ds003626-2.1.2:$PROJECT_ROOT/ds005170-1.1.2:$PROJECT_ROOT/data/external:/data:/scratch:/mnt:/home/$USER"
export MAX_DEPTH=7
bash "$SCRIPT_DIR/01_scan_datasets.sh"
echo ""

# ──────────────────────────────────────────────────────────────────
# STEP 2: Build unified trials.csv manifest
# ──────────────────────────────────────────────────────────────────
echo "[STEP 2/4] Preparing Unified Metadata (trials.csv)..."
echo "[PHASE 1/3] Metadata assembly and label resolution"
python "$SCRIPT_DIR/build_manifests.py"
python "$SCRIPT_DIR/fix_trials.py"

TRIALS_CSV="$PROJECT_ROOT/data/processed/manifests/trials.csv"
export TRIALS_CSV

python - <<'PY'
import os
from pathlib import Path
import pandas as pd

trials_csv = Path(os.environ["TRIALS_CSV"])
if trials_csv.exists():
  df = pd.read_csv(trials_csv)
  label_col = "label" if "label" in df.columns else ("trial_type" if "trial_type" in df.columns else "task")
  print("[INFO] Label summary by dataset:")
  for dataset, sub in df.groupby("dataset"):
    counts = sub[label_col].astype(str).value_counts()
    print(f"    {dataset}: {len(counts)} classes via '{label_col}'")
    print(counts.head(8).to_string())
PY

# Quick validation
if [[ -f "$TRIALS_CSV" ]]; then
  TRIAL_COUNT=$(wc -l < "$TRIALS_CSV")
  echo "[INFO] trials.csv has $TRIAL_COUNT lines (including header)"
  echo "[INFO] Dataset breakdown:"
  cut -d',' -f4 "$TRIALS_CSV" | tail -n +2 | sort | uniq -c || true
else
  echo "[ERROR] trials.csv not found at $TRIALS_CSV! Aborting."
  exit 1
fi
echo ""

# ──────────────────────────────────────────────────────────────────
# STEP 3: Parallel Preprocessing & Caching
# ──────────────────────────────────────────────────────────────────
if [[ "$SKIP_PREPROCESS" == "0" ]]; then
  echo "[PHASE 2/3] Parallel preprocessing and caching"
  echo "[STEP 3/4] Parallel Preprocessing & Caching..."
  echo "This step uses joblib multiprocessing to process all datasets concurrently."
  echo "An automatic CPU OOM guardrail limits max workers to (CPU_CORES / 2)."
  PREPROCESS_JOBS="${PREPROCESS_JOBS:-8}"
  echo "[INFO] PREPROCESS_JOBS=${PREPROCESS_JOBS} (override with PREPROCESS_JOBS=<n>)"
  python "$SCRIPT_DIR/03_preprocess_and_cache.py" \
    --project_root "$PROJECT_ROOT" \
    --target_sfreq 250.0 \
    --jobs "$PREPROCESS_JOBS" \
    2>&1 | tee "$PROJECT_ROOT/master_preprocessing.log"
  echo ""
else
  echo "[STEP 3/4] Skipping preprocessing (--skip-preprocess)"
  echo ""
fi

# ──────────────────────────────────────────────────────────────────
# STEP 4: Sequential Training
# ──────────────────────────────────────────────────────────────────
echo "[PHASE 3/3] Sequential training"
echo "[STEP 4/4] Sequential Training..."
echo "Training datasets sequentially to prevent GPU Out-of-Memory (OOM) crashes."

# Iterate over supported datasets
for DATASET in chisco thinking_out_loud zuco; do
    echo "----------------------------------------------------"
    echo "  Starting Training: $DATASET"
    echo "----------------------------------------------------"

    if [[ "$DATASET" == "zuco" ]]; then
      echo "[INFO] ZuCo is pretraining-only; skipping supervised training in this pipeline."
      continue
    fi
    
    # Check if this dataset has any trials in the manifest
    if grep -q ",$DATASET," "$TRIALS_CSV" 2>/dev/null; then
      bash "$SCRIPT_DIR/04_train.sh" "$DATASET" "$EXPERIMENT_NAME" foreground \
        2>&1 | tee "$PROJECT_ROOT/master_training_$DATASET.log"
      echo "Finished training $DATASET!"
    else
      echo "[SKIP] No trials found for $DATASET in trials.csv"
    fi
done

echo "==========================================================="
echo "                 PIPELINE COMPLETED                        "

# ──────────────────────────────────────────────────────────────────
# STAGE 4 (Optional): RL Fine-tuning
# ──────────────────────────────────────────────────────────────────
RUN_RL_STAGE="${RUN_RL_STAGE:-0}"

if [[ "$RUN_RL_STAGE" == "1" ]]; then
  echo ""
  echo "==========================================================="
  echo "[STAGE 4/4] RL Fine-tuning with SCST Rewards"
  echo "==========================================================="
  
  # RL fine-tuning on each trained model
  for DATASET in chisco thinking_out_loud; do
    if [[ "$DATASET" == "zuco" ]]; then
      continue
    fi
    
    STAGE3_CKPT="$PROJECT_ROOT/experiments/$EXPERIMENT_NAME/best_model.pt"
    if [[ -f "$STAGE3_CKPT" ]]; then
      echo ""
      echo "  [RL] Starting Stage 4 fine-tuning for $DATASET..."
      echo "  [RL] Using Stage 3 checkpoint: $STAGE3_CKPT"
      
      python "$SCRIPT_DIR/train_with_rl.py" \
        --config configs/models/conformer_small.yaml \
        --dataset "$DATASET" \
        --experiment "${EXPERIMENT_NAME}_RL_stage4" \
        --stage 4 \
        --pretrained "$STAGE3_CKPT" \
        --rl-config configs/training/train_rl.yaml \
        2>&1 | tee "$PROJECT_ROOT/master_rl_finetuning_$DATASET.log"
      
      echo "  [RL] Finished RL stage 4 for $DATASET!"
    else
      echo "[SKIP-RL] No Stage 3 checkpoint found for $DATASET at $STAGE3_CKPT"
    fi
  done
  
  echo ""
  echo "==========================================================="
  echo "       STAGE 4 RL FINE-TUNING COMPLETED"
  echo "==========================================================="
else
  echo ""
  echo "[INFO] Stage 4 RL fine-tuning skipped. To enable, run:"
  echo "       RUN_RL_STAGE=1 bash scripts/05_run_all_end_to_end.sh ..."
  echo ""
fi

echo "==========================================================="
echo "            FULL PIPELINE COMPLETED                        "
echo "==========================================================="
echo "==========================================================="
