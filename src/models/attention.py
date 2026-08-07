"""
Attention modules — Multi-head self-attention with relative positional encoding.

Implements Shaw et al. 2018 relative position bias for the Conformer encoder.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional relative positional encoding.

    When ``pos_emb`` is passed to ``forward()``, relative position biases
    are computed and added to the raw attention logits before softmax.

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout applied to attention weights.
    max_rel_pos : int
        Maximum relative distance for clipping (Shaw et al. 2018).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        max_rel_pos: int = 128,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.max_rel_pos = max_rel_pos

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Relative position bias table (Shaw et al. 2018)
        # Total positions = 2 * max_rel_pos + 1 (negative, zero, positive)
        n_positions = 2 * max_rel_pos + 1
        self.rel_pos_bias = nn.Embedding(n_positions, n_heads)
        # Initialize with small values
        nn.init.normal_(self.rel_pos_bias.weight, std=0.02)

    def _compute_relative_position_bias(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """Compute relative position bias matrix.

        Parameters
        ----------
        seq_len : int
            Sequence length.
        device : torch.device
            Target device.

        Returns
        -------
        torch.Tensor
            Bias matrix, shape (1, n_heads, seq_len, seq_len).
        """
        # positions[i, j] = i - j (relative distance from query i to key j)
        positions = torch.arange(seq_len, device=device)
        rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)  # (L, L)

        # Clip to [-max_rel_pos, max_rel_pos] and shift to [0, 2*max_rel_pos]
        rel_pos = rel_pos.clamp(-self.max_rel_pos, self.max_rel_pos) + self.max_rel_pos

        # Look up bias: (L, L, n_heads) → (n_heads, L, L)
        bias = self.rel_pos_bias(rel_pos)  # (L, L, n_heads)
        bias = bias.permute(2, 0, 1).unsqueeze(0)  # (1, n_heads, L, L)

        return bias

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        use_rel_pos: bool = True,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch, seq_len, d_model).
        mask : torch.Tensor, optional
            Padding mask, shape (batch, seq_len). True = valid.
        use_rel_pos : bool
            Whether to add relative positional bias to attention.

        Returns
        -------
        torch.Tensor
            Output, shape (batch, seq_len, d_model).
        """
        batch, seq_len, d = x.shape

        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, d_head)
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative positional bias
        if use_rel_pos:
            rel_bias = self._compute_relative_position_bias(seq_len, x.device)
            attn = attn + rel_bias

        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq)
            attn = attn.masked_fill(~attn_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(batch, seq_len, -1)
        return self.out_proj(out)


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (added to input embeddings)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return positional encoding for input length.

        Parameters
        ----------
        x : torch.Tensor
            Input, shape (batch, seq_len, d_model).

        Returns
        -------
        torch.Tensor
            Positional encoding, shape (1, seq_len, d_model).
        """
        return self.pe[:, : x.size(1)]


class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding marker.

    When use_relative_pos=True, the Conformer uses learned relative position
    biases inside MultiHeadSelfAttention (Shaw et al. 2018) rather than
    additive sinusoidal embeddings. This module adds sinusoidal encodings to
    the input as a secondary signal but the primary relative bias is computed
    inside the attention layer.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]
