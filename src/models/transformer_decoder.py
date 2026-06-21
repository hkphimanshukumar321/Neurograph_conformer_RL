"""
Transformer Decoder — autoregressive text generation from EEG encoder output.

Standard Transformer decoder with:
  - Masked multi-head self-attention
  - Cross-attention to encoder output
  - Feed-forward network
  - CTC auxiliary branch
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerDecoder(nn.Module):
    """Autoregressive Transformer decoder for EEG-to-text generation.

    Parameters
    ----------
    vocab_size : int
        Output vocabulary size.
    d_model : int
        Model dimension (must match encoder d_model).
    n_layers : int
        Number of decoder layers.
    n_heads : int
        Number of attention heads.
    d_ff : int
        Feed-forward inner dimension.
    dropout : float
        Dropout rate.
    max_seq_len : int
        Maximum output sequence length.
    tie_embeddings : bool
        Whether to tie input and output embeddings.
    ctc_weight : float
        Weight for CTC auxiliary loss branch. 0 to disable.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 50,
        tie_embeddings: bool = True,
        ctc_weight: float = 0.3,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Token embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        self.emb_scale = math.sqrt(d_model)

        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.output_proj.weight = self.token_emb.weight

        # CTC branch (projects encoder output to vocab)
        self.ctc_weight = ctc_weight
        if ctc_weight > 0:
            self.ctc_proj = nn.Linear(d_model, vocab_size)
        else:
            self.ctc_proj = None

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass (teacher forcing).

        Parameters
        ----------
        tgt : torch.Tensor
            Target token IDs, shape (batch, tgt_len).
        memory : torch.Tensor
            Encoder output, shape (batch, src_len, d_model).
        tgt_mask : torch.Tensor, optional
            Causal mask for target.
        memory_mask : torch.Tensor, optional
            Mask for encoder output.

        Returns
        -------
        torch.Tensor
            Logits, shape (batch, tgt_len, vocab_size).
        """
        batch, tgt_len = tgt.shape

        # Token + positional embedding
        positions = torch.arange(tgt_len, device=tgt.device).unsqueeze(0)
        x = self.token_emb(tgt) * self.emb_scale + self.pos_emb(positions)
        x = self.emb_dropout(x)

        # Generate causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self._generate_causal_mask(tgt_len, device=tgt.device)

        # Decoder layers
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

        x = self.final_norm(x)
        logits = self.output_proj(x)  # (batch, tgt_len, vocab_size)

        return logits

    def ctc_logits(self, memory: torch.Tensor) -> torch.Tensor:
        """Compute CTC logits from encoder output.

        Parameters
        ----------
        memory : torch.Tensor
            Encoder output, shape (batch, src_len, d_model).

        Returns
        -------
        torch.Tensor
            CTC logits, shape (batch, src_len, vocab_size).
        """
        if self.ctc_proj is None:
            raise RuntimeError("CTC branch is disabled (ctc_weight=0)")
        return self.ctc_proj(memory)

    @torch.no_grad()
    def greedy_decode(
        self,
        memory: torch.Tensor,
        sos_id: int,
        eos_id: int,
        max_len: int | None = None,
    ) -> torch.Tensor:
        """Greedy autoregressive decoding.

        Parameters
        ----------
        memory : torch.Tensor
            Encoder output, shape (batch, src_len, d_model).
        sos_id : int
            Start-of-sequence token ID.
        eos_id : int
            End-of-sequence token ID.
        max_len : int, optional
            Maximum decoding length. Defaults to self.max_seq_len.

        Returns
        -------
        torch.Tensor
            Generated token IDs, shape (batch, generated_len).
        """
        if max_len is None:
            max_len = self.max_seq_len

        batch = memory.size(0)
        device = memory.device

        generated = torch.full((batch, 1), sos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logits = self.forward(generated, memory)  # (batch, cur_len, vocab)
            next_token = logits[:, -1, :].argmax(dim=-1)  # (batch,)

            # Mask finished sequences
            next_token = next_token.masked_fill(finished, eos_id)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

            finished = finished | (next_token == eos_id)
            if finished.all():
                break

        return generated

    def sample_decode(
        self,
        memory: torch.Tensor,
        sos_id: int,
        eos_id: int,
        temperature: float = 1.0,
        max_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample-based decoding (for SCST).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (generated_ids, log_probs) — both shape (batch, generated_len).
        """
        if max_len is None:
            max_len = self.max_seq_len

        batch = memory.size(0)
        device = memory.device

        generated = torch.full((batch, 1), sos_id, dtype=torch.long, device=device)
        log_probs_list = []
        finished = torch.zeros(batch, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logits = self.forward(generated, memory)
            logits = logits[:, -1, :] / temperature  # (batch, vocab)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (batch,)
            log_prob = F.log_softmax(logits, dim=-1)
            selected_log_prob = log_prob.gather(1, next_token.unsqueeze(1)).squeeze(1)

            next_token = next_token.masked_fill(finished, eos_id)
            selected_log_prob = selected_log_prob.masked_fill(finished, 0.0)

            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            log_probs_list.append(selected_log_prob)

            finished = finished | (next_token == eos_id)
            if finished.all():
                break

        log_probs = torch.stack(log_probs_list, dim=1)  # (batch, gen_len)
        return generated, log_probs

    @staticmethod
    def _generate_causal_mask(size: int, device: torch.device) -> torch.Tensor:
        """Generate upper-triangular causal mask."""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask.bool()  # True = masked positions


class DecoderLayer(nn.Module):
    """Single Transformer decoder layer."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()

        # Masked self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Masked self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = residual + self.dropout(x)

        # Cross-attention
        residual = x
        x = self.norm2(x)
        x, _ = self.cross_attn(x, memory, memory, key_padding_mask=memory_mask)
        x = residual + self.dropout(x)

        # FFN
        residual = x
        x = self.norm3(x)
        x = residual + self.ffn(x)

        return x
