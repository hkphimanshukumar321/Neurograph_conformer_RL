#!/bin/bash
# Master Script: Run all training phases sequentially
set -e

DATASET=${1:-"chisco"}
PROTOCOL=${2:-"within_subject"}

echo "======================================"
echo " Starting Full Training Pipeline"
echo " Dataset: $DATASET"
echo " Protocol: $PROTOCOL"
echo "======================================"

# Run Phase 1
bash scripts/train_phase1.sh $DATASET $PROTOCOL

# Run Phase 2
bash scripts/train_phase2.sh $DATASET $PROTOCOL

# Run Phase 3
bash scripts/train_phase3.sh $DATASET $PROTOCOL

echo "======================================"
echo " All Training Phases Completed Successfully!"
echo " Check 'experiments/phase3_${DATASET}' for the final model."
echo "======================================"
