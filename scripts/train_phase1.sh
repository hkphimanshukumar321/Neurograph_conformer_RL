#!/bin/bash
# Phase 1: Pre-train Encoder
# 
# Uses neurograph_full (d_model=256) so that Phase 2/3 checkpoints
# are dimensionally compatible — all phases share the same architecture.
# Decoder is disabled (classification-only phase).
set -e

DATASET=${1:-"chisco"}
PROTOCOL=${2:-"within_subject"}
EXP_NAME="phase1_${DATASET}"

echo "======================================"
echo "Starting Phase 1: Pre-training Encoder"
echo "Dataset: $DATASET"
echo "Experiment: $EXP_NAME"
echo "======================================"

python scripts/train.py \
    --config configs/models/neurograph_full.yaml \
    --dataset $DATASET \
    --protocol $PROTOCOL \
    --experiment $EXP_NAME \
    --overrides \
        "training.loss_weights.cls=1.0" \
        "training.loss_weights.contrast=0.3" \
        "training.loss_weights.gen=0.0" \
        "training.loss_weights.rl=0.0" \
        "training.batch_size=32" \
        "training.lr=5e-4" \
        "training.max_epochs=200" \
        "training.early_stopping.patience=30" \
        "training.label_smoothing=0.1" \
        "model.encoder.dropout=0.3" \
        "model.decoder.enabled=false"

echo "Phase 1 Complete."
