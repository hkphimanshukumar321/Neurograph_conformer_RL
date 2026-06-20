#!/bin/bash
# Phase 3: RL SCST Fine-tuning
set -e

DATASET=${1:-"chisco"}
PROTOCOL=${2:-"within_subject"}
EXP_NAME="phase3_${DATASET}"
PRETRAINED="experiments/phase2_${DATASET}/checkpoints/best.pt"

echo "======================================"
echo "Starting Phase 3: RL SCST Fine-tuning"
echo "Dataset: $DATASET"
echo "Experiment: $EXP_NAME"
echo "Loading pretrained from: $PRETRAINED"
echo "======================================"

python scripts/train.py \
    --config configs/models/neurograph_full.yaml \
    --dataset $DATASET \
    --protocol $PROTOCOL \
    --experiment $EXP_NAME \
    --pretrained $PRETRAINED \
    --overrides "training.loss_weights.cls=0.0" "training.loss_weights.contrast=0.0" "training.loss_weights.gen=0.0" "training.loss_weights.rl=1.0" "model.decoder.enabled=true" "training.lr=0.00001"

echo "Phase 3 Complete."
