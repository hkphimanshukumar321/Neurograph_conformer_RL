#!/bin/bash
# 04_train.sh - Starts the training loop in the background.

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <dataset_name> [experiment_name]"
    echo "Example: $0 chisco First_Run"
    exit 1
fi

DATASET=$1
EXPERIMENT=${2:-"Default_Run"}
MODE=${3:-"background"}

# Start training in the background using nohup
echo "[INFO] Starting training for dataset: $DATASET (Experiment: $EXPERIMENT)"
echo "[INFO] Logs will be written to training_$DATASET.log"

if [ "$MODE" = "foreground" ]; then
    echo "[INFO] Running training in foreground. Press Ctrl+C to abort."
    python scripts/train.py \
            python scripts/train_with_rl.py \
        --config configs/models/conformer_small.yaml \
        --dataset "$DATASET" \
        --experiment "$EXPERIMENT" \
            --stage 3 \
        2>&1 | tee "training_$DATASET.log"
else
    nohup python scripts/train.py \
            nohup python scripts/train_with_rl.py \
            --stage 3 \
        --config configs/models/conformer_small.yaml \
        --dataset "$DATASET" \
        --experiment "$EXPERIMENT" \
        > "training_$DATASET.log" 2>&1 &
    
    echo "[SUCCESS] Training started in background. Process ID: $!"
    echo "To view live logs, run: tail -f training_$DATASET.log"
fi
