"""
Conformer Encoder — combines convolution (local) with self-attention (global).

Based on: Gulati et al., "Conformer: Convolution-augmented Transformer for
Speech Recognition", 2020.

Adapted for EEG temporal sequence modeling with relative positional encoding.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.attention import (
    MultiHeadSelfAttention,
    RelativePositionalEncoding,
    SinusoidalPositionalEncoding,
)


class ConformerEncoder(nn.Module):
    """Stack of Conformer blocks for temporal modeling.

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_layers : int
        Number of Conformer blocks.
    n_heads : int
        Number of attention heads.
    d_ff : int
        Feed-forward inner dimension.
    conv_kernel_size : int
        Kernel size for the convolution module.
    dropout : float
        Dropout rate.
    attention_dropout : float
        Attention-specific dropout rate.
    use_relative_pos : bool
        Whether to use relative positional encoding.
    macaron : bool
        Whether to use Macaron-style half-step feed-forward.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 512,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_relative_pos: bool = True,
        macaron: bool = True,
    ):
        super().__init__()
        self.d_model = d_model

        # Positional encoding
        self.pos_encoding = (
            RelativePositionalEncoding(d_model)
            if use_relative_pos
            else SinusoidalPositionalEncoding(d_model)
        )

        self.layers = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
                attention_dropout=attention_dropout,
                macaron=macaron,
            )
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through Conformer encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence, shape (batch, L, d_model).
        mask : torch.Tensor, optional
            Padding mask, shape (batch, L). True = valid, False = padding.

        Returns
        -------
        torch.Tensor
            Encoded sequence, shape (batch, L, d_model).
        """
        # Add positional encoding
        pos_emb = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, pos_emb=pos_emb, mask=mask)

        return self.final_norm(x)


class ConformerBlock(nn.Module):
    """Single Conformer block.

    Structure (Macaron-style):
        x → FFN(½) → MHSA → Conv → FFN(½) → LayerNorm → output

    Each sub-module has residual connections and layer normalization.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        conv_kernel_size: int,
        dropout: float,
        attention_dropout: float,
        macaron: bool = True,
    ):
        super().__init__()
        self.macaron = macaron

        # First half-step FFN (Macaron-style)
        if macaron:
            self.ffn1 = FeedForwardModule(d_model, d_ff, dropout)
            self.ffn1_norm = nn.LayerNorm(d_model)
            self.ffn1_scale = 0.5
        else:
            self.ffn1_scale = 0.0

        # Multi-Head Self-Attention
        self.mhsa = MultiHeadSelfAttention(
            d_model, n_heads, dropout=attention_dropout
        )
        self.mhsa_norm = nn.LayerNorm(d_model)

        # Convolution module
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.conv_norm = nn.LayerNorm(d_model)

        # Second FFN
        self.ffn2 = FeedForwardModule(d_model, d_ff, dropout)
        self.ffn2_norm = nn.LayerNorm(d_model)
        self.ffn2_scale = 0.5 if macaron else 1.0

        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input, shape (batch, L, d_model).
        pos_emb : torch.Tensor, optional
            Positional embedding.
        mask : torch.Tensor, optional
            Padding mask.

        Returns
        -------
        torch.Tensor
            Output, shape (batch, L, d_model).
        """
        # First FFN (half-step, Macaron)
        if self.macaron:
            residual = x
            x = self.ffn1_norm(x)
            x = residual + self.ffn1_scale * self.dropout(self.ffn1(x))

        # MHSA
        residual = x
        x = self.mhsa_norm(x)
        x = residual + self.dropout(self.mhsa(x, mask=mask))

        # Convolution
        residual = x
        x = self.conv_norm(x)
        x = residual + self.dropout(self.conv(x))

        # Second FFN (half-step if Macaron)
        residual = x
        x = self.ffn2_norm(x)
        x = residual + self.ffn2_scale * self.dropout(self.ffn2(x))

        return self.final_norm(x)


class FeedForwardModule(nn.Module):
    """Conformer feed-forward module with Swish activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),  # Swish
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvolutionModule(nn.Module):
    """Conformer convolution module.

    Pointwise Conv → GLU → Depthwise Conv → BatchNorm → Swish → Pointwise Conv
    """

    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"

        self.layer_norm = nn.LayerNorm(d_model)

        self.net = nn.Sequential(
            # Pointwise expansion (2x for GLU)
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1),
            nn.GLU(dim=1),
            # Depthwise conv
            nn.Conv1d(
                d_model, d_model,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=d_model,  # depthwise
            ),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),  # Swish
            # Pointwise projection
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input, shape (batch, L, d_model).

        Returns
        -------
        torch.Tensor
            Output, shape (batch, L, d_model).
        """
        x = x.transpose(1, 2)  # (batch, d_model, L)
        x = self.net(x)
        return x.transpose(1, 2)  # (batch, L, d_model)



