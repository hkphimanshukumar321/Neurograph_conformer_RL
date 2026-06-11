.PHONY: help install dev test lint preprocess train evaluate ablation export profile clean

PYTHON ?= python
PIP ?= pip

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ──── Setup ────

install:  ## Install package in production mode
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

dev:  ## Install package in development mode with dev dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev,deploy]"
	pre-commit install

# ──── Data Pipeline ────

scan:  ## Scan the raw datasets directory and generate index manifest
	bash scripts/01_scan_datasets.sh

preprocess:  ## Run unified preprocessing on all datasets
	$(PYTHON) scripts/03_preprocess_and_cache.py --project_root .

splits:  ## Build train/val/test splits
	$(PYTHON) scripts/build_splits.py --manifest data/manifests/dataset_manifest.json \
		--protocols within_subject,loso --output data/splits

# ──── Training ────

train-baseline:  ## Train EEGNet baseline on Thinking Out Loud (within-subject)
	$(PYTHON) scripts/train.py \
		--config configs/models/baselines/eegnet.yaml \
		--dataset thinking_out_loud --protocol within_subject --experiment E1_baseline

train-proposed:  ## Train proposed Conformer model on Thinking Out Loud (LOSO)
	$(PYTHON) scripts/train.py \
		--config configs/models/neurograph_full.yaml \
		--dataset thinking_out_loud --protocol loso --experiment E1_proposed

pretrain:  ## Stage 1: Self-supervised pretraining on ZuCo
	$(PYTHON) scripts/train.py --config configs/training/pretrain_ssl.yaml \
		--dataset zuco --experiment E2_pretrain

train-seq:  ## Stage 3: Sequence training on Chisco
	$(PYTHON) scripts/train.py --config configs/training/train_seq.yaml \
		--dataset chisco --experiment E6_sequence

train-rl:  ## Stage 4: RL fine-tuning on Chisco
	$(PYTHON) scripts/train.py --config configs/training/train_rl.yaml \
		--dataset chisco --experiment E6_rl

# ──── Evaluation ────

evaluate:  ## Evaluate best checkpoint
	$(PYTHON) scripts/evaluate.py --checkpoint results/checkpoints/best.pt \
		--dataset thinking_out_loud --protocol loso --metrics all

ablation:  ## Run full ablation suite
	$(PYTHON) scripts/ablation/run_ablation.py --config configs/training/train_cls.yaml \
		--dataset thinking_out_loud --ablations A1,A2,A3,A4,A5,A6,A7,A8,A9

# ──── Deployment ────

export:  ## Export model to ONNX with INT8 quantization
	$(PYTHON) scripts/export_model.py --checkpoint results/checkpoints/best.pt \
		--format onnx --quantize int8 --output results/deployment/model.onnx

profile:  ## Profile model inference latency and memory
	$(PYTHON) scripts/profile_model.py --checkpoint results/checkpoints/best.pt \
		--variants small,medium,full --devices cpu,cuda

# ──── Quality ────

test:  ## Run unit tests
	pytest tests/ -v --tb=short

lint:  ## Run linter
	ruff check src/ scripts/ tests/

format:  ## Format code
	ruff format src/ scripts/ tests/

# ──── Cleanup ────

clean:  ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/
