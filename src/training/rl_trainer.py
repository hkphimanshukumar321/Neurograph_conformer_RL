"""
SCST (Self-Critical Sequence Training) and RL fine-tuning trainer.

Stage 4: Fine-tune the generation decoder using sequence-level rewards.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.losses import SCSTLoss, MWERLoss

logger = logging.getLogger(__name__)


class RewardComputer:
    """Multi-dimensional reward computation for RL fine-tuning.

    Combines semantic similarity, WER, CER, retrieval correctness,
    fluency, and phoneme consistency into a single scalar reward.

    Parameters
    ----------
    reward_cfg : dict
        Reward weights and settings.
    """

    def __init__(self, reward_cfg: dict):
        self.weights = {}
        for key in ["semantic_similarity", "wer", "cer", "retrieval_correctness",
                     "fluency", "phoneme_consistency"]:
            weight_key = f"{key}_weight"
            if weight_key in reward_cfg:
                self.weights[key] = reward_cfg.get(weight_key, 0.0)
            elif key in reward_cfg and isinstance(reward_cfg[key], dict):
                self.weights[key] = reward_cfg[key].get("weight", 0.0)

        # Lazy-load semantic similarity encoder
        self._sem_encoder = None
        self._sem_model_name = reward_cfg.get(
            "semantic_similarity", {}
        ).get("encoder", "paraphrase-multilingual-MiniLM-L12-v2")

    def _get_semantic_encoder(self):
        if self._sem_encoder is None:
            from sentence_transformers import SentenceTransformer
            # Force CPU to avoid CUDA library version mismatches (e.g. libcudart.so.13)
            # in containers. Reward computation is on short strings so CPU is fast enough.
            self._sem_encoder = SentenceTransformer(self._sem_model_name, device="cpu")
        return self._sem_encoder

    def compute(
        self,
        predictions: list[str],
        references: list[str],
    ) -> torch.Tensor:
        """Compute composite reward.

        Parameters
        ----------
        predictions : list[str]
            Generated text sequences.
        references : list[str]
            Reference text sequences.

        Returns
        -------
        torch.Tensor
            Reward values, shape (batch,).
        """
        batch_size = len(predictions)
        rewards = torch.zeros(batch_size)

        # Semantic similarity
        if self.weights.get("semantic_similarity", 0) > 0:
            try:
                encoder = self._get_semantic_encoder()
                pred_emb = encoder.encode(predictions, convert_to_tensor=True)
                ref_emb = encoder.encode(references, convert_to_tensor=True)
                sim = F.cosine_similarity(pred_emb, ref_emb)
                rewards += self.weights["semantic_similarity"] * sim.cpu()
            except Exception as e:
                logger.warning(f"Semantic similarity failed: {e}")

        # WER
        if self.weights.get("wer", 0) > 0:
            try:
                from jiwer import wer
                for i, (pred, ref) in enumerate(zip(predictions, references)):
                    w = wer(ref, pred)
                    rewards[i] += self.weights["wer"] * (1.0 - w)  # reward = 1 - WER
            except ImportError:
                pass

        # CER
        if self.weights.get("cer", 0) > 0:
            try:
                from jiwer import cer
                for i, (pred, ref) in enumerate(zip(predictions, references)):
                    c = cer(ref, pred)
                    rewards[i] += self.weights["cer"] * (1.0 - c)
            except ImportError:
                pass

        return rewards


class RLTrainer:
    """RL fine-tuning trainer for Stage 4.

    Parameters
    ----------
    model : nn.Module
        The model with generation head.
    cfg : dict
        RL training configuration.
    device : str
        Device to use.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: dict,
        device: str = "auto",
    ):
        self.model = model
        self.cfg = cfg
        self.device = device

        method = cfg.get("method", "scst")
        if method == "scst":
            self.loss_fn = SCSTLoss()
        elif method == "mwer":
            self.loss_fn = MWERLoss(
                n_best=cfg.get("mwer", {}).get("n_best", 5),
            )
        else:
            raise ValueError(f"Unknown RL method: {method}")

        self.reward_computer = RewardComputer(cfg.get("reward", {}))

        logger.info(f"RLTrainer initialized with method={method}")

    def train_step(
        self,
        encoder_output: torch.Tensor,
        reference_texts: list[str],
        tokenizer: Any,
    ) -> dict[str, float]:
        """Single RL training step.

        Parameters
        ----------
        encoder_output : torch.Tensor
            Encoder output from the model, shape (batch, seq_len, d_model).
        reference_texts : list[str]
            Ground truth text sequences.
        tokenizer : Any
            Tokenizer for decoding predictions.

        Returns
        -------
        dict[str, float]
            Training metrics.
        """
        self.model.train()

        decoder = self.model.generation_head.decoder

        # Greedy decode (baseline)
        greedy_ids = decoder.greedy_decode(
            encoder_output,
            sos_id=tokenizer.SOS_ID,
            eos_id=tokenizer.EOS_ID,
        )

        # Sample decode
        sample_ids, sample_log_probs = decoder.sample_decode(
            encoder_output,
            sos_id=tokenizer.SOS_ID,
            eos_id=tokenizer.EOS_ID,
            temperature=1.0,
        )

        # Decode to text
        greedy_texts = [tokenizer.decode(ids) for ids in greedy_ids]
        sample_texts = [tokenizer.decode(ids) for ids in sample_ids]

        # Compute rewards
        greedy_rewards = self.reward_computer.compute(greedy_texts, reference_texts)
        sample_rewards = self.reward_computer.compute(sample_texts, reference_texts)

        # SCST loss
        loss = self.loss_fn(
            sample_log_probs,
            sample_rewards.to(self.device),
            greedy_rewards.to(self.device),
        )

        return {
            "rl_loss": loss.item(),
            "rl_loss_tensor": loss,
            "greedy_reward": greedy_rewards.mean().item(),
            "sample_reward": sample_rewards.mean().item(),
            "advantage": (sample_rewards - greedy_rewards).mean().item(),
        }
