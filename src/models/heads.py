"""
Output heads — Classification, Retrieval, and Generation heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """Linear classification head with optional dropout.

    Parameters
    ----------
    d_model : int
        Input feature dimension.
    n_classes : int
        Number of output classes.
    dropout : float
        Dropout rate before classifier.
    pooling : str
        How to pool sequence → single vector: 'mean', 'cls', 'max'.
    """

    def __init__(
        self,
        d_model: int,
        n_classes: int,
        dropout: float = 0.3,
        pooling: str = "mean",
    ):
        super().__init__()
        self.pooling = pooling
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Encoder output, shape (batch, L, d_model).
        mask : torch.Tensor, optional
            Padding mask, shape (batch, L). True = valid.

        Returns
        -------
        torch.Tensor
            Class logits, shape (batch, n_classes).
        """
        # Pool sequence to single vector
        if self.pooling == "mean":
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()  # (batch, L, 1)
                x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                x = x.mean(dim=1)
        elif self.pooling == "cls":
            x = x[:, 0, :]  # first token
        elif self.pooling == "max":
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            x = x.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return self.head(x)


class RetrievalHead(nn.Module):
    """Contrastive retrieval head — projects EEG to semantic embedding space.

    Parameters
    ----------
    d_model : int
        Input feature dimension from encoder.
    d_embed : int
        Semantic embedding dimension (must match text encoder dimension).
    temperature : float
        Temperature for InfoNCE contrastive loss.
    """

    def __init__(
        self,
        d_model: int,
        d_embed: int = 256,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_embed),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project encoder output to semantic embedding space.

        Parameters
        ----------
        x : torch.Tensor
            Encoder output, shape (batch, L, d_model).
        mask : torch.Tensor, optional
            Padding mask.

        Returns
        -------
        torch.Tensor
            Semantic embedding, shape (batch, d_embed). L2-normalized.
        """
        # Mean pooling
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)

        projected = self.projection(pooled)

        # L2 normalize for cosine similarity
        return F.normalize(projected, p=2, dim=-1)

    def compute_similarity(
        self,
        eeg_emb: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute similarity matrix between EEG and text embeddings.

        Parameters
        ----------
        eeg_emb : torch.Tensor
            EEG embeddings, shape (N, d_embed).
        text_emb : torch.Tensor
            Text embeddings, shape (M, d_embed).

        Returns
        -------
        torch.Tensor
            Similarity matrix, shape (N, M).
        """
        return torch.mm(eeg_emb, text_emb.t()) / self.temperature


class GenerationHead(nn.Module):
    """Wrapper that combines encoder output routing with the Transformer decoder.

    This head manages:
    - Routing encoder output to the decoder
    - Teacher forcing ratio scheduling
    - CTC + CE loss computation
    """

    def __init__(
        self,
        decoder: nn.Module,
        d_model: int,
    ):
        super().__init__()
        self.decoder = decoder
        self.d_model = d_model

    def forward(
        self,
        encoder_output: torch.Tensor,
        tgt_tokens: torch.Tensor,
        encoder_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for training.

        Parameters
        ----------
        encoder_output : torch.Tensor
            Encoder output, shape (batch, src_len, d_model).
        tgt_tokens : torch.Tensor
            Target token IDs (shifted right), shape (batch, tgt_len).
        encoder_mask : torch.Tensor, optional
            Encoder output mask.

        Returns
        -------
        dict with:
            'logits': (batch, tgt_len, vocab_size)
            'ctc_logits': (batch, src_len, vocab_size) if CTC enabled
        """
        result = {}

        # Autoregressive decoder
        logits = self.decoder(tgt_tokens, encoder_output, memory_mask=encoder_mask)
        result["logits"] = logits

        # CTC branch
        if self.decoder.ctc_proj is not None:
            result["ctc_logits"] = self.decoder.ctc_logits(encoder_output)

        return result
