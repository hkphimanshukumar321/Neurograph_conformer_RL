#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <DATASET> <TASK> <SEED> <GPU_ID>"
    echo "Example: $0 karaone phoneme 42 0"
    exit 1
fi

DATASET=$1
TASK=$2
SEED=$3
GPU_ID=$4

VARIANTS=(
    "baseline"
    "ghost_only"
    "dwaspp_only"
    "cas_only"
    "ghost_dwaspp"
    "ghost_cas"
    "dwaspp_cas"
    "full_ghost_dwaspp_cas"
    "normalconv_dwaspp_cas"
)

export CUDA_VISIBLE_DEVICES=$GPU_ID

for VARIANT in "${VARIANTS[@]}"; do
    echo "=========================================================="
    echo "Running Ablation: $VARIANT on $DATASET ($TASK) | Seed $SEED"
    echo "=========================================================="
    
    OUT_DIR="results/ablations/frontend/${DATASET}/${TASK}/${VARIANT}/seed_${SEED}"
    mkdir -p "$OUT_DIR"
    
    # TODO: REPLACE WITH ACTUAL TRAIN COMMAND
    # Assuming Hydra or OmegaConf based CLI where we can merge the variant config
    # Example: python -m src.training.train dataset=$DATASET task=$TASK seed=$SEED \
    #          model/ablation=frontend/${VARIANT} output_dir=$OUT_DIR
    
    echo "Executing training for $VARIANT..."
    # Placeholder execution
    python -m src.training.train \
        --dataset "$DATASET" \
        --task "$TASK" \
        --seed "$SEED" \
        --output_dir "$OUT_DIR" \
        --ablation_config "configs/ablations/frontend/${VARIANT}.yaml" \
        > "${OUT_DIR}/train.log" 2>&1 || echo "Warning: Training script failed or not implemented yet."
        
    echo "Finished $VARIANT. Logs saved to ${OUT_DIR}/train.log"
done

echo "=========================================================="
echo "All frontend ablations completed for dataset $DATASET, task $TASK."
echo "You can now run:"
echo "python scripts/aggregate_frontend_ablation.py --dataset $DATASET --task $TASK --seed $SEED"
echo "=========================================================="
