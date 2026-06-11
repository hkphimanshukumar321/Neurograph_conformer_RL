#!/bin/bash
# 05_run_all_end_to_end.sh - Master script to run the entire NeuroGraph-Conformer pipeline.
#
# This script executes:
# 1. Dataset scanning
# 2. Metadata extraction
# 3. Parallel Preprocessing & Caching (CPU-bound)
# 4. Sequential Training (GPU-bound, sequential to avoid OOM)

set -e # Exit immediately if a command exits with a non-zero status

echo "==========================================================="
echo "   NeuroGraph-Conformer End-to-End Pipeline Initialization "
echo "==========================================================="

PROJECT_ROOT=$(pwd)
EXPERIMENT_NAME=${1:-"End_to_End_Run"}

echo "[STEP 1/4] Scanning Server Datasets..."
export PROJECT_ROOT
export SEARCH_ROOTS="/data:/scratch:/mnt:/home/$USER"
export MAX_DEPTH=7
bash scripts/01_scan_datasets.sh
echo ""

echo "[STEP 2/4] Preparing Unified Metadata (trials.csv)..."
bash scripts/02_prepare_metadata.sh
echo ""

echo "[STEP 3/4] Parallel Preprocessing & Caching..."
echo "This step uses joblib multiprocessing to process all datasets concurrently."
echo "An automatic CPU OOM guardrail limits max workers to (CPU_CORES / 2)."
python scripts/03_preprocess_and_cache.py --target_sfreq 250.0 --jobs -1
echo ""

echo "[STEP 4/4] Sequential Training..."
echo "Training datasets sequentially to prevent GPU Out-of-Memory (OOM) crashes."

# Iterate over supported datasets
# Note: Kara One is deprecated/excluded in the current codebase logic.
for DATASET in chisco zuco thinking_out_loud; do
    echo "----------------------------------------------------"
    echo "  Starting Training: $DATASET"
    echo "----------------------------------------------------"
    
    # Run in foreground to block the loop until finished
    bash scripts/04_train.sh "$DATASET" "$EXPERIMENT_NAME" foreground
    
    echo "Finished training $DATASET!"
done

echo "==========================================================="
echo "                 PIPELINE COMPLETED                        "
echo "==========================================================="
