"""
run_hpo.py — Run Optuna hyperparameter optimization.

Usage:
    python scripts/run_hpo.py \
        --config configs/models/conformer_medium.yaml \
        --dataset kara_one \
        --n-trials 100 \
        --search-space full \
        --wandb-project NeuroGraph-Conformer-RL
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config, merge_configs
from src.utils.logging import setup_logger
from src.utils.seed import seed_everything
from src.training.hpo import OptunaHPO, build_objective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Optimization")
    parser.add_argument("--config", type=str, required=True, help="Base model config")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["kara_one", "thinking_out_loud", "chisco", "zuco"])
    parser.add_argument("--n-trials", type=int, default=100, help="Number of trials")
    parser.add_argument("--search-space", type=str, default="full",
                        choices=["full", "training_only", "quick"],
                        help="Search space type")
    parser.add_argument("--study-name", type=str, default=None,
                        help="Optuna study name (default: auto)")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL (e.g., sqlite:///optuna.db)")
    parser.add_argument("--pruner", type=str, default="median",
                        choices=["median", "hyperband", "none"])
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel workers")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Time limit in seconds")
    parser.add_argument("--wandb-project", type=str, default="NeuroGraph-Conformer-RL",
                        help="W&B project name")
    parser.add_argument("--output", type=str, default="results/hpo",
                        help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("hpo", level="INFO")
    seed_everything(args.seed)

    # Load configs
    base_cfg = load_config("configs/base.yaml")
    dataset_cfg = load_config(f"configs/datasets/{args.dataset}.yaml")
    model_cfg = load_config(args.config)
    cfg = merge_configs(base_cfg, dataset_cfg, model_cfg)

    study_name = args.study_name or f"hpo_{args.dataset}_{args.search_space}"

    logger.info(f"Starting HPO: study={study_name}, dataset={args.dataset}")
    logger.info(f"Search space: {args.search_space}, n_trials={args.n_trials}")

    # Dummy training function (replace with real training)
    def train_fn(trial_cfg, trial):
        """Placeholder training function.

        In production, this would:
        1. Build dataset and dataloaders
        2. Build model from trial_cfg
        3. Train for N epochs
        4. Return validation balanced accuracy
        """
        import numpy as np

        # Simulate training with random metric for now
        # Replace this with actual model training
        logger.info(f"Trial {trial.number}: training with config...")

        # Report intermediate values for pruning
        for epoch in range(10):
            val_acc = np.random.uniform(0.1, 0.5) + epoch * 0.02
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise __import__("optuna").TrialPruned()

        return val_acc

    # Build objective
    objective = build_objective(cfg, train_fn, search_space=args.search_space)

    # Storage
    storage = args.storage
    if storage is None:
        output_dir = Path(args.output) / study_name
        output_dir.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{output_dir / 'optuna.db'}"

    # Create and run HPO
    hpo = OptunaHPO(
        study_name=study_name,
        direction="maximize",
        n_trials=args.n_trials,
        storage=storage,
        pruner=args.pruner,
        wandb_project=args.wandb_project,
    )

    study = hpo.run(objective, n_jobs=args.n_jobs, timeout=args.timeout)

    # Save results
    output_dir = Path(args.output) / study_name
    hpo.save_results(output_dir)

    # Save best config
    best_cfg = hpo.get_best_config(cfg)
    from src.utils.config import save_config
    save_config(best_cfg, output_dir / "best_config.yaml")

    logger.info(f"HPO complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
