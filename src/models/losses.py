"""
Loss functions — all losses used across training stages.

Includes classification, contrastive, CTC, retrieval, and RL losses.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance (Lin et al., 2017).

    Parameters
    ----------
    gamma : float
        Focusing parameter. Higher = more focus on hard examples.
    alpha : torch.Tensor or None
        Per-class weights.
    label_smoothing : float
        Label smoothing factor.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, targets, weight=self.alpha,
            label_smoothing=self.label_smoothing, reduction="none"
        )
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss for EEG-text alignment.

    Parameters
    ----------
    temperature : float
        Temperature scaling.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        eeg_emb: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute symmetric InfoNCE loss.

        Parameters
        ----------
        eeg_emb : torch.Tensor
            L2-normalized EEG embeddings, shape (batch, d_embed).
        text_emb : torch.Tensor
            L2-normalized text embeddings, shape (batch, d_embed).

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        # Cosine similarity matrix
        sim = torch.mm(eeg_emb, text_emb.t()) / self.temperature  # (B, B)
        labels = torch.arange(sim.size(0), device=sim.device)

        # Symmetric loss
        loss_eeg = F.cross_entropy(sim, labels)
        loss_text = F.cross_entropy(sim.t(), labels)

        return (loss_eeg + loss_text) / 2


class NTXentLoss(nn.Module):
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) for self-supervised learning.

    Used for EEG augmentation contrastive learning: two augmented views of
    the same EEG trial should have similar representations.

    Parameters
    ----------
    temperature : float
        Temperature scaling.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute NT-Xent loss between two augmented views.

        Parameters
        ----------
        z1, z2 : torch.Tensor
            L2-normalized embeddings from two augmentation views,
            shape (batch, d_embed).

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        batch = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # (2B, d)
        sim = torch.mm(z, z.t()) / self.temperature  # (2B, 2B)

        # Mask out self-similarity
        mask = torch.eye(2 * batch, device=z.device).bool()
        sim.masked_fill_(mask, float("-inf"))

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(batch, 2 * batch, device=z.device),
            torch.arange(0, batch, device=z.device),
        ])

        return F.cross_entropy(sim, labels)


class TripletContrastiveLoss(nn.Module):
    """Triplet margin loss for retrieval.

    Parameters
    ----------
    margin : float
        Margin for triplet loss.
    """

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        return self.loss_fn(anchor, positive, negative)


class CTCLoss(nn.Module):
    """CTC loss wrapper for alignment-free sequence prediction.

    Parameters
    ----------
    blank_id : int
        Blank token ID for CTC.
    """

    def __init__(self, blank_id: int = 0):
        super().__init__()
        self.loss_fn = nn.CTCLoss(blank=blank_id, reduction="mean", zero_infinity=True)

    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss.

        Parameters
        ----------
        log_probs : torch.Tensor
            Log probabilities, shape (T, batch, vocab_size).
            Note: CTC expects time-first ordering.
        targets : torch.Tensor
            Target sequences (concatenated), shape (sum(target_lengths),).
        input_lengths : torch.Tensor
            Length of each input sequence, shape (batch,).
        target_lengths : torch.Tensor
            Length of each target sequence, shape (batch,).
        """
        return self.loss_fn(log_probs, targets, input_lengths, target_lengths)


class SCSTLoss(nn.Module):
    """Self-Critical Sequence Training (SCST) loss.

    Uses greedy decode as baseline to reduce variance in REINFORCE.

    R_scst = -(R(sample) - R(greedy)) * log P(sample)
    """

    def forward(
        self,
        sample_log_probs: torch.Tensor,
        sample_rewards: torch.Tensor,
        greedy_rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SCST loss.

        Parameters
        ----------
        sample_log_probs : torch.Tensor
            Log probabilities of sampled sequences, shape (batch, seq_len).
        sample_rewards : torch.Tensor
            Reward for sampled sequences, shape (batch,).
        greedy_rewards : torch.Tensor
            Reward for greedy sequences (baseline), shape (batch,).

        Returns
        -------
        torch.Tensor
            Scalar loss (to be minimized).
        """
        advantage = sample_rewards - greedy_rewards  # (batch,)

        # REINFORCE with baseline
        # Negative because we minimize loss but want to maximize reward
        loss = -(advantage.unsqueeze(1) * sample_log_probs).sum(dim=1).mean()

        return loss


class MWERLoss(nn.Module):
    """Minimum Word Error Rate (MWER) loss.

    Samples N-best hypotheses and weights by WER.
    """

    def __init__(self, n_best: int = 5, smoothing: float = 0.01):
        super().__init__()
        self.n_best = n_best
        self.smoothing = smoothing

    def forward(
        self,
        nbest_log_probs: torch.Tensor,
        nbest_wers: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MWER loss.

        Parameters
        ----------
        nbest_log_probs : torch.Tensor
            Log probabilities of N-best, shape (batch, n_best).
        nbest_wers : torch.Tensor
            WER for each hypothesis, shape (batch, n_best).

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        # Normalize probabilities within N-best
        probs = F.softmax(nbest_log_probs, dim=1)  # (batch, n_best)

        # Expected WER
        expected_wer = (probs * nbest_wers).sum(dim=1)  # (batch,)

        return expected_wer.mean()
