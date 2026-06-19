"""
run_hpo.py — Run Optuna hyperparameter optimization.

Usage:
    python scripts/hpo/run_hpo.py \
        --config configs/models/conformer_small.yaml \
        --dataset chisco \
        --n-trials 50 \
        --search-space quick \
        --wandb-project NeuroGraph-Conformer-RL
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import shutil
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Suppress sklearn warnings during heavy HPO
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.*")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import load_config, merge_configs, config_to_dict
from src.utils.logging import setup_logger
from src.utils.seed import seed_everything
from src.training.hpo import OptunaHPO, build_objective, OptunaPruningCallback
from src.datasets.factory import get_dataset
from src.models.baselines import BASELINE_REGISTRY
from src.models.neurograph import NeuroGraphConformer
from src.training.trainer import Trainer


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
    parser.add_argument("--device", type=str, default="auto")
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

    # Prepare dataset early since it's the same for all trials
    n_channels = cfg.dataset.get("n_channels_original", 64)
    n_samples = int(cfg.preprocessing.get("target_srate", 250) *
                    cfg.preprocessing.get("epoch_tmax", 2.0))

    project_root = Path(__file__).resolve().parent.parent.parent
    manifest_path = project_root / "data" / "processed" / "manifests" / "trials.csv"

    if not manifest_path.exists():
        logger.error(f"Manifest not found at {manifest_path}. Run preprocessing first.")
        return

    full_dataset = get_dataset(
        args.dataset,
        manifest_path=manifest_path,
        split="train",
        max_samples=n_samples,
    )

    if len(full_dataset) == 0:
        logger.error(f"No samples found for {args.dataset} in {manifest_path}")
        return

    # Split 80/20
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    logger.info(f"Loaded {train_size} training samples, {val_size} validation samples.")
    n_classes = full_dataset.n_classes

    # Define the actual training function
    def train_fn(trial_cfg, trial):
        """Train the model for a given trial configuration."""
        
        # Datasets are pre-loaded, just create dataloaders with new batch size
        batch_size = trial_cfg.training.get("batch_size", 32)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Build model from trial_cfg
        model_name = trial_cfg.model.name
        if model_name in BASELINE_REGISTRY:
            model = BASELINE_REGISTRY[model_name](trial_cfg.model)
        else:
            model = NeuroGraphConformer(
                cfg=trial_cfg.model,
                n_channels=n_channels,
                n_samples=n_samples,
                n_classes=n_classes,
            )

        # We need a temporary dir for the trainer's checkpoints so we don't clobber
        temp_dir = tempfile.mkdtemp(prefix=f"hpo_trial_{trial.number}_")
        
        try:
            # We don't initialize a full WandbLogger here, because Optuna's
            # WeightsAndBiasesCallback already handles logging to W&B for HPO.
            trainer = Trainer(
                model=model,
                cfg=trial_cfg,
                train_loader=train_loader,
                val_loader=val_loader,
                device=args.device,
                experiment_dir=temp_dir,
                logger_obj=None, # HPO handles W&B via callback
            )
            
            # We inject the Optuna pruning callback into the trainer manually
            # since Trainer doesn't natively support callbacks yet.
            # We wrap the validate method.
            pruning_cb = OptunaPruningCallback(trial, monitor="val_balanced_accuracy")
            
            # Monkey-patch fit to include pruning
            original_fit = trainer.fit
            
            def fit_with_pruning():
                cfg_train = trainer.cfg.training
                max_epochs = cfg_train.max_epochs
                patience = cfg_train.early_stopping.get("patience", 30)
                monitor = cfg_train.early_stopping.get("monitor", "val_balanced_accuracy")
                mode = cfg_train.early_stopping.get("mode", "max")

                logger.info(f"[Trial {trial.number}] Starting training: {max_epochs} epochs")

                for epoch in range(max_epochs):
                    trainer.current_epoch = epoch
                    train_metrics = trainer.train_epoch()
                    val_metrics = trainer.validate()

                    if trainer.scheduler is not None:
                        trainer.scheduler.step()

                    # Trigger Optuna pruning callback
                    pruning_cb(epoch, val_metrics)
                    
                    # Early stopping logic
                    current_metric = val_metrics[monitor]
                    improved = (mode == "max" and current_metric > trainer.best_metric) or \
                               (mode == "min" and current_metric < trainer.best_metric)

                    if improved:
                        trainer.best_metric = current_metric
                        trainer.patience_counter = 0
                    else:
                        trainer.patience_counter += 1
                        if trainer.patience_counter >= patience:
                            logger.info(f"[Trial {trial.number}] Early stopping at epoch {epoch+1}")
                            break
                            
                return {"val_balanced_accuracy": trainer.best_metric}
                
            # Run the patched training loop
            result = fit_with_pruning()
            return result["val_balanced_accuracy"]
            
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

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
