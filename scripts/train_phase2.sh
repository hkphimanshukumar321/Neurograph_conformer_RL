#!/bin/bash
# Phase 2: Supervised Generation
set -e

DATASET=${1:-"chisco"}
PROTOCOL=${2:-"within_subject"}
EXP_NAME="phase2_${DATASET}"
PRETRAINED="experiments/phase1_${DATASET}/checkpoints/best.pt"

echo "======================================"
echo "Starting Phase 2: Supervised Generation"
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
    --overrides "training.loss_weights.cls=0.1" "training.loss_weights.contrast=0.1" "training.loss_weights.gen=1.0" "training.loss_weights.rl=0.0" "model.decoder.enabled=true" "training.lr=0.0001"

echo "Phase 2 Complete."
