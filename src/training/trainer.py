"""
Main Trainer — orchestrates multi-task training across all stages.

Supports:
  - Classification (CE / Focal loss)
  - Contrastive retrieval (SupCon loss on retrieval embeddings)
  - Generation (CTC + CE — Phase 2, gated by config)
  - RL fine-tuning (SCST — Phase 3, gated by config)

Each loss component is weighted by ``cfg.training.loss_weights`` and all
metrics are tracked in a unified history dict for graphical output.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.models.losses import SupConLoss
from src.utils.device import DeviceManager
from src.utils.seed import seed_everything
from src.datasets.tokenizer import Tokenizer
from src.training.rl_trainer import RLTrainer

logger = logging.getLogger(__name__)


class Trainer:
    """Multi-task trainer for NeuroGraph-Conformer.

    Parameters
    ----------
    model : nn.Module
        The model to train (should be NeuroGraphConformer or a baseline).
    cfg : DictConfig
        Full merged configuration.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    device : str
        Device to train on.
    experiment_dir : str or Path
        Directory for logs, checkpoints, and plots.
    logger_obj : Any, optional
        W&B or other external logger.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: DictConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "auto",
        experiment_dir: str | Path = "experiments/default",
        logger_obj: Any | None = None,
        tokenizer: Tokenizer | None = None,
    ):
        self.cfg = cfg
        self.device_mgr = DeviceManager(device)
        self.device = self.device_mgr.device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.logger_obj = logger_obj

        # ── Loss weights from config ──
        lw = cfg.training.get("loss_weights", {})
        self.loss_weights = {
            "cls": float(lw.get("cls", 1.0)),
            "contrast": float(lw.get("contrast", 0.3)),
            "ctc": float(lw.get("ctc", 0.3)),
            "gen": float(lw.get("gen", 1.0)),
            "rl": float(lw.get("rl", 0.1)),
        }

        # ── Detect active heads ──
        self.has_retrieval = hasattr(model, "retrieval_head") and model.retrieval_head is not None
        self.has_generation = hasattr(model, "generation_head") and model.generation_head is not None

        # ── Build losses & trainers ──
        self.cls_criterion = self._build_cls_criterion()
        self.contrast_criterion = self._build_contrast_criterion()
        
        self.gen_criterion = None
        self.ctc_criterion = None
        if self.has_generation and (self.loss_weights["gen"] > 0 or self.loss_weights["ctc"] > 0):
            self.gen_criterion = nn.CrossEntropyLoss(ignore_index=0) # 0 is PAD_ID
            self.ctc_criterion = nn.CTCLoss(blank=0, zero_infinity=True)

        self.rl_trainer = None
        if self.has_generation and self.loss_weights["rl"] > 0:
            rl_cfg = self.cfg.get("model", {}).get("rl", {})
            self.rl_trainer = RLTrainer(self.model, rl_cfg, self.device)

        # ── Optimizer ──
        self.optimizer = self._build_optimizer()

        # ── Scheduler ──
        self.scheduler = self._build_scheduler()

        # ── State ──
        self.current_epoch = 0
        self.best_metric = 0.0
        self.patience_counter = 0

        # Log active components
        active = ["classification"]
        if self.has_retrieval and self.loss_weights["contrast"] > 0:
            active.append(f"contrastive(w={self.loss_weights['contrast']})")
        if self.has_generation and self.loss_weights["gen"] > 0:
            active.append(f"generation(w={self.loss_weights['gen']})")
        if self.has_generation and self.loss_weights["rl"] > 0:
            active.append(f"rl(w={self.loss_weights['rl']})")
        logger.info(f"Trainer initialized on {self.device_mgr.summary()}")
        logger.info(f"Active training objectives: {', '.join(active)}")
        logger.info(f"Loss weights: {self.loss_weights}")

    # ──────────────────────────────────────────────────────────────
    # Loss builders
    # ──────────────────────────────────────────────────────────────

    def _build_cls_criterion(self) -> nn.Module:
        """Build classification loss (CE with label smoothing)."""
        label_smoothing = self.cfg.training.get("label_smoothing", 0.1)
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def _build_contrast_criterion(self) -> nn.Module:
        """Build supervised contrastive loss for retrieval embeddings."""
        temp = 0.07
        if hasattr(self.cfg, "model"):
            heads_cfg = self.cfg.model.get("heads", self.cfg.model.get("arch", {}).get("heads", {}))
            ret_cfg = heads_cfg.get("retrieval", {})
            temp = ret_cfg.get("temperature", 0.07)
        return SupConLoss(temperature=temp)

    # ──────────────────────────────────────────────────────────────
    # Optimizer & scheduler
    # ──────────────────────────────────────────────────────────────

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer from config."""
        cfg = self.cfg.training
        opt_name = cfg.get("optimizer", "adamw").lower()

        if hasattr(self.model, "get_param_groups"):
            params = self.model.get_param_groups(cfg.lr, cfg.weight_decay)
        else:
            params = self.model.parameters()

        if opt_name == "adamw":
            return torch.optim.AdamW(
                params, lr=cfg.lr, weight_decay=cfg.weight_decay,
                betas=(0.9, 0.999),
            )
        elif opt_name == "adam":
            return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        elif opt_name == "sgd":
            return torch.optim.SGD(
                params, lr=cfg.lr, weight_decay=cfg.weight_decay,
                momentum=0.9, nesterov=True,
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

    def _build_scheduler(self) -> Any:
        """Build learning rate scheduler."""
        cfg = self.cfg.training
        sched_name = cfg.get("scheduler", "cosine_warmup")

        if sched_name == "cosine_warmup":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.max_epochs - cfg.get("warmup_epochs", 10),
                eta_min=cfg.get("min_lr", 1e-6),
            )
        elif sched_name == "constant_with_warmup":
            return torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
        else:
            return None

    # ──────────────────────────────────────────────────────────────
    # Training loop
    # ──────────────────────────────────────────────────────────────

    def train_epoch(self) -> dict[str, float]:
        """Train for one epoch with multi-task objectives.

        Returns
        -------
        dict[str, float]
            Training metrics including per-component losses.
        """
        self.model.train()

        total_loss = 0.0
        total_cls_loss = 0.0
        total_contrast_loss = 0.0
        total_gen_loss = 0.0
        total_rl_loss = 0.0
        total_greedy_reward = 0.0
        total_sample_reward = 0.0
        total_advantage = 0.0
        correct = 0
        total = 0

        from tqdm import tqdm
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1} Train",
            leave=False,
        )

        for batch_idx, batch in enumerate(pbar):
            eeg = batch["eeg"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # Phase 2 needs target tokens
            target_tokens = None
            if "target_tokens" in batch and self.has_generation:
                target_tokens = batch["target_tokens"].to(self.device)

            # ── Multi-task forward pass ──
            # task="all" activates classification + retrieval + generation (if target_tokens provided)
            use_task = "all" if self.has_retrieval else "classification"
            outputs = self.model(eeg, task=use_task, tgt_tokens=target_tokens)

            # ── Classification loss ──
            cls_logits = outputs["cls_logits"]
            cls_loss = self.cls_criterion(cls_logits, labels)

            # ── Contrastive loss (SupCon on retrieval embeddings) ──
            contrast_loss = torch.tensor(0.0, device=self.device)
            if self.has_retrieval and "retrieval_emb" in outputs and self.loss_weights["contrast"] > 0:
                retrieval_emb = outputs["retrieval_emb"]  # (B, d_embed), L2-normalized
                contrast_loss = self.contrast_criterion(retrieval_emb, labels)

            # ── Generation & CTC Loss (Phase 2) ──
            gen_loss = torch.tensor(0.0, device=self.device)
            ctc_loss = torch.tensor(0.0, device=self.device)
            
            target_lengths = batch.get("target_length", torch.tensor(0)).to(self.device)
            has_text_in_batch = target_lengths.sum() > 0
            
            if self.has_generation and has_text_in_batch and target_tokens is not None:
                if self.loss_weights["gen"] > 0 and "logits" in outputs:
                    gen_logits = outputs["logits"] # (B, T, V)
                    # CE expects (B, C, d1, d2) -> (B, V, T)
                    ce_loss = self.gen_criterion(gen_logits.transpose(1, 2), target_tokens)
                    gen_loss += ce_loss
                    
                if self.loss_weights["ctc"] > 0 and "ctc_logits" in outputs:
                    ctc_logits = outputs["ctc_logits"] # (B, src_len, V)
                    B, src_len, V = ctc_logits.shape
                    # CTC expects (src_len, B, V) log_probs
                    log_probs = torch.nn.functional.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
                    input_lengths = torch.full((B,), src_len, dtype=torch.long, device=self.device)
                    # For CTC, target_tokens shouldn't have SOS/EOS, but ManifestDataset adds EOS. We leave it as is.
                    c_loss = self.ctc_criterion(log_probs, target_tokens, input_lengths, target_lengths)
                    ctc_loss += c_loss

            # ── RL Loss (Phase 3) ──
            rl_loss_tensor = torch.tensor(0.0, device=self.device)
            rl_metrics = {}
            if self.rl_trainer is not None and self.loss_weights["rl"] > 0 and self.tokenizer is not None and has_text_in_batch:
                texts = batch["text"]
                # Pass encoder_output instead of raw eeg to save compute
                rl_metrics = self.rl_trainer.train_step(outputs["encoder_output"], texts, self.tokenizer)
                pass 

            # ── Weighted total loss ──
            loss = (
                self.loss_weights["cls"] * cls_loss
                + self.loss_weights["contrast"] * contrast_loss
                + self.loss_weights["gen"] * gen_loss
                + self.loss_weights["ctc"] * ctc_loss
            )
            
            # If RL is active, we just add it to loss if we patched it, otherwise we do a separate step.
            if "rl_loss_tensor" in rl_metrics:
                loss += self.loss_weights["rl"] * rl_metrics["rl_loss_tensor"]

            # ── Backward ──
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            clip_val = self.cfg.training.get("gradient_clip_val", 1.0)
            if clip_val > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), clip_val)

            self.optimizer.step()

            # ── Accumulate metrics ──
            bs = eeg.size(0)
            total_loss += loss.item() * bs
            total_cls_loss += cls_loss.item() * bs
            total_contrast_loss += contrast_loss.item() * bs
            total_gen_loss += gen_loss.item() * bs
            if self.loss_weights["ctc"] > 0:
                # Add to a counter if needed, here we just accumulate total_loss
                pass
            if "rl_loss" in rl_metrics:
                total_rl_loss += rl_metrics["rl_loss"] * bs
                total_greedy_reward += rl_metrics.get("greedy_reward", 0.0) * bs
                total_sample_reward += rl_metrics.get("sample_reward", 0.0) * bs
                total_advantage += rl_metrics.get("advantage", 0.0) * bs
                
            preds = cls_logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += bs

            postfix_dict = {
                "loss": f"{loss.item():.4f}",
                "cls": f"{cls_loss.item():.4f}",
            }
            if self.loss_weights["contrast"] > 0:
                postfix_dict["con"] = f"{contrast_loss.item():.4f}"
            if self.loss_weights["gen"] > 0:
                postfix_dict["gen"] = f"{gen_loss.item():.4f}"
            if self.loss_weights["rl"] > 0 and "rl_loss" in rl_metrics:
                postfix_dict["rl"] = f"{rl_metrics['rl_loss']:.4f}"
                
            pbar.set_postfix(postfix_dict)

        metrics = {
            "train_loss": total_loss / total,
            "train_cls_loss": total_cls_loss / total,
            "train_contrast_loss": total_contrast_loss / total,
            "train_accuracy": correct / total,
        }
        if self.loss_weights["gen"] > 0:
            metrics["train_gen_loss"] = total_gen_loss / total
        if self.loss_weights["rl"] > 0:
            metrics["train_rl_loss"] = total_rl_loss / total
            metrics["rl_loss"] = total_rl_loss / total
            metrics["rl_greedy_reward"] = total_greedy_reward / total
            metrics["rl_sample_reward"] = total_sample_reward / total
            metrics["rl_advantage"] = total_advantage / total
            
        return metrics

    # ──────────────────────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self, loader: DataLoader | None = None, prefix: str = "val", desc: str | None = None) -> dict[str, float]:
        """Run validation or testing with multi-task metrics.

        Parameters
        ----------
        loader : DataLoader, optional
            Data loader to evaluate. Defaults to self.val_loader.
        prefix : str
            Prefix for metric keys (e.g., 'val', 'test').
        desc : str, optional
            Description for progress bar.

        Returns
        -------
        dict[str, float]
            Evaluation metrics.
        """
        self.model.eval()
        if loader is None:
            loader = self.val_loader
            
        total_loss = 0.0
        total_cls_loss = 0.0
        total_contrast_loss = 0.0
        all_preds = []
        all_labels = []
        all_embeddings = []

        from tqdm import tqdm
        
        if desc is None:
            desc = f"Epoch {self.current_epoch + 1} {prefix.capitalize()}"
            
        pbar = tqdm(
            loader,
            desc=desc,
            leave=False,
        )

        for batch in pbar:
            eeg = batch["eeg"].to(self.device)
            labels = batch["label"].to(self.device)

            use_task = "all" if self.has_retrieval else "classification"
            outputs = self.model(eeg, task=use_task)
            logits = outputs["cls_logits"]

            # Classification loss
            cls_loss = self.cls_criterion(logits, labels)

            # Contrastive loss
            contrast_loss = torch.tensor(0.0, device=self.device)
            if self.has_retrieval and "retrieval_emb" in outputs and self.loss_weights["contrast"] > 0:
                contrast_loss = self.contrast_criterion(outputs["retrieval_emb"], labels)
                all_embeddings.append(outputs["retrieval_emb"].cpu())

            loss = (
                self.loss_weights["cls"] * cls_loss
                + self.loss_weights["contrast"] * contrast_loss
            )

            # Phase 2: Generation validation
            gen_loss = torch.tensor(0.0, device=self.device)
            target_tokens = None
            target_lengths = batch.get("target_length", torch.tensor(0)).to(self.device)
            has_text_in_batch = target_lengths.sum() > 0
            
            if "target_tokens" in batch and self.has_generation and self.loss_weights["gen"] > 0 and has_text_in_batch:
                target_tokens = batch["target_tokens"].to(self.device)
                outputs = self.model(eeg, task=use_task, tgt_tokens=target_tokens)
                if "logits" in outputs:
                    gen_logits = outputs["logits"]
                    ce_loss = self.gen_criterion(gen_logits.transpose(1, 2), target_tokens)
                    gen_loss += ce_loss
                    loss += self.loss_weights["gen"] * gen_loss

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_cls_loss += cls_loss.item() * bs
            total_contrast_loss += contrast_loss.item() * bs

            all_preds.append(logits.argmax(dim=-1).cpu())
            all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        n = len(all_labels)
        accuracy = (all_preds == all_labels).float().mean().item()

        # Balanced accuracy and F1
        import warnings
        from sklearn.metrics import balanced_accuracy_score, f1_score

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            balanced_acc = balanced_accuracy_score(all_labels.numpy(), all_preds.numpy())
            macro_f1 = f1_score(
                all_labels.numpy(), all_preds.numpy(),
                average="macro", zero_division=0,
            )

        # Embedding quality: silhouette score (measures cluster separation)
        silhouette = 0.0
        if all_embeddings:
            try:
                from sklearn.metrics import silhouette_score
                emb_np = torch.cat(all_embeddings).numpy()
                lab_np = all_labels.numpy()
                n_unique = len(np.unique(lab_np))
                if n_unique >= 2 and len(lab_np) > n_unique:
                    silhouette = silhouette_score(
                        emb_np, lab_np,
                        metric="cosine",
                        sample_size=min(2000, len(lab_np)),
                    )
            except Exception as e:
                logger.debug(f"Silhouette score computation failed: {e}")

        # Store raw predictions for confusion matrix generation
        if prefix == "val":
            self._last_val_preds = all_preds.numpy()
            self._last_val_labels = all_labels.numpy()
        elif prefix == "test":
            self._last_test_preds = all_preds.numpy()
            self._last_test_labels = all_labels.numpy()

        metrics = {
            f"{prefix}_loss": total_loss / n,
            f"{prefix}_cls_loss": total_cls_loss / n,
            f"{prefix}_contrast_loss": total_contrast_loss / n,
            f"{prefix}_accuracy": accuracy,
            f"{prefix}_balanced_accuracy": balanced_acc,
            f"{prefix}_macro_f1": macro_f1,
            f"{prefix}_silhouette_score": silhouette,
        }
        return metrics

    # ──────────────────────────────────────────────────────────────
    # Full training loop
    # ──────────────────────────────────────────────────────────────

    def fit(self) -> dict[str, list]:
        """Run full training loop.

        Returns
        -------
        dict[str, list]
            History of all metrics per epoch, including per-component losses
            and embedding quality — ready for graphical output.
        """
        cfg = self.cfg.training
        max_epochs = cfg.max_epochs
        patience = cfg.early_stopping.get("patience", 30)
        monitor = cfg.early_stopping.get("monitor", "val_balanced_accuracy")
        mode = cfg.early_stopping.get("mode", "max")

        # ── History tracks EVERYTHING for plotting ──
        history = {
            # Per-epoch timing
            "lr": [],
            "epoch_time_s": [],
            # Training losses (per component)
            "train_loss": [],
            "train_cls_loss": [],
            "train_contrast_loss": [],
            "train_accuracy": [],
            # Validation losses (per component)
            "val_loss": [],
            "val_cls_loss": [],
            "val_contrast_loss": [],
            # Validation metrics
            "val_accuracy": [],
            "val_balanced_accuracy": [],
            "val_macro_f1": [],
            "val_silhouette_score": [],
            # RL metrics (populated in Phase 3 when RL is active)
            "rl_greedy_reward": [],
            "rl_sample_reward": [],
            "rl_advantage": [],
            "rl_loss": [],
        }

        logger.info(f"Starting training: {max_epochs} epochs, patience={patience}")
        logger.info(f"Monitoring: {monitor} (mode={mode})")

        for epoch in range(max_epochs):
            self.current_epoch = epoch
            t_start = time.time()

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            elapsed = time.time() - t_start
            lr = self.optimizer.param_groups[0]["lr"]

            # ── Record history ──
            for k in history:
                if k in train_metrics:
                    history[k].append(train_metrics[k])
                elif k in val_metrics:
                    history[k].append(val_metrics[k])
                elif k == "lr":
                    history[k].append(lr)
                elif k == "epoch_time_s":
                    history[k].append(elapsed)
                # RL metrics stay empty until Phase 3

            # ── Log ──
            log_parts = [
                f"Epoch {epoch+1}/{max_epochs}",
                f"loss={train_metrics['train_loss']:.4f}",
                f"cls={train_metrics['train_cls_loss']:.4f}",
            ]
            if self.has_retrieval and self.loss_weights["contrast"] > 0:
                log_parts.append(f"con={train_metrics['train_contrast_loss']:.4f}")
            log_parts.extend([
                f"val_bacc={val_metrics['val_balanced_accuracy']:.4f}",
                f"val_f1={val_metrics['val_macro_f1']:.4f}",
            ])
            if self.has_retrieval:
                log_parts.append(f"sil={val_metrics['val_silhouette_score']:.3f}")
            log_parts.extend([f"lr={lr:.2e}", f"{elapsed:.1f}s"])

            logger.info(" | ".join(log_parts))

            if self.logger_obj is not None:
                self.logger_obj.log_epoch(epoch + 1, train_metrics, val_metrics, lr)

            # ── Early stopping ──
            current_metric = val_metrics[monitor]
            improved = (mode == "max" and current_metric > self.best_metric) or \
                       (mode == "min" and current_metric < self.best_metric)

            if improved:
                self.best_metric = current_metric
                self.patience_counter = 0
                self._save_checkpoint("best.pt", val_metrics)
                logger.info(f"  ✓ New best {monitor}: {self.best_metric:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1} (patience={patience})")
                    break

        # ── Save final checkpoint ──
        self._save_checkpoint("last.pt", val_metrics)

        # ── Save confusion matrix data ──
        if hasattr(self, "_last_val_preds"):
            np.savez(
                self.experiment_dir / "val_predictions.npz",
                preds=self._last_val_preds,
                labels=self._last_val_labels,
            )
            logger.info(f"Saved val predictions to {self.experiment_dir / 'val_predictions.npz'}")

        return history

    # ──────────────────────────────────────────────────────────────
    # Checkpointing
    # ──────────────────────────────────────────────────────────────

    def _save_checkpoint(self, filename: str, metrics: dict) -> None:
        """Save model checkpoint."""
        ckpt_dir = self.experiment_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "metrics": metrics,
            "config": dict(self.cfg),
        }

        torch.save(checkpoint, ckpt_dir / filename)
        logger.info(f"Saved checkpoint: {ckpt_dir / filename}")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_metric = checkpoint["best_metric"]
        logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")
