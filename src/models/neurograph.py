"""
NeuroGraph-Conformer — Full model assembly.

Composes: Frontend → Graph Spatial Encoder → Conformer → (Mamba) → Heads

This is the main model class that wires all components together based on config.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.conformer import ConformerEncoder
from src.models.eeg_lightweight_frontend import (
    EEGGhostDWASPPCASFrontEnd,
    NodeAttentionPool,
)
from src.models.frontend import FilterBankFrontEnd, WaveletFrontEnd
from src.models.graph_encoder import GraphSpatialEncoder
from src.models.heads import ClassificationHead, GenerationHead, RetrievalHead
from src.models.mamba_module import MambaModule
from src.models.transformer_decoder import TransformerDecoder

logger = logging.getLogger(__name__)


class NeuroGraphConformer(nn.Module):
    """NeuroGraph-Conformer: Multi-level EEG neural language decoding model.

    Parameters
    ----------
    cfg : DictConfig
        Model configuration (from configs/models/*.yaml).
    n_channels : int
        Number of input EEG channels (or regions if using region pooling).
    n_samples : int
        Number of time samples per epoch.
    n_classes : int or dict
        Number of classes per task. If dict: {'T1': 11, 'T2': 4, ...}.
    vocab_size : int, optional
        Output vocabulary size for generation head.
    """

    def __init__(
        self,
        cfg: DictConfig,
        n_channels: int = 9,
        n_samples: int = 500,
        n_classes: int | dict = 11,
        vocab_size: int | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        arch = cfg.get("arch", cfg)  # handle nested config
        d_model = arch.get("encoder", {}).get("d_model", 128)

        # ──── 1. Front-End ────
        frontend_cfg = arch.get("frontend", {})
        frontend_type = frontend_cfg.get("type", "raw")

        if frontend_type == "cwt":
            self.frontend = WaveletFrontEnd(
                n_freqs=frontend_cfg.get("freqs_max", 100) - frontend_cfg.get("freqs_min", 1) + 1,
                freq_min=frontend_cfg.get("freqs_min", 1),
                freq_max=frontend_cfg.get("freqs_max", 100),
                sfreq=250.0,
                n_cycles_mode=frontend_cfg.get("n_cycles_mode", "adaptive"),
                time_downsample=frontend_cfg.get("time_downsample", 4),
                d_out=frontend_cfg.get("projection_dim", 0),
            )
            # After CWT: (batch, C, F, T') or (batch, C, d_out)
            if frontend_cfg.get("projection_dim", 0) > 0:
                frontend_out_dim = frontend_cfg["projection_dim"]
            else:
                n_freqs = frontend_cfg.get("freqs_max", 100)
                t_out = n_samples // frontend_cfg.get("time_downsample", 4)
                frontend_out_dim = n_freqs * t_out
            # Frequency bins available for lightweight frontend
            self._cwt_n_freqs = frontend_cfg.get("freqs_max", 100) - frontend_cfg.get("freqs_min", 1) + 1
        elif frontend_type == "filterbank":
            self.frontend = FilterBankFrontEnd(
                n_bands=frontend_cfg.get("n_bands", 6),
                sfreq=250.0,
                d_out=frontend_cfg.get("d_out", d_model),
            )
            frontend_out_dim = frontend_cfg.get("d_out", d_model)
            self._cwt_n_freqs = frontend_cfg.get("n_bands", 6)
        else:
            # Raw EEG — use a simple linear projection
            self.frontend = nn.Linear(n_samples, d_model)
            frontend_out_dim = d_model
            self._cwt_n_freqs = 0

        # ──── 1b. Lightweight EEG Front-End (Ghost-DWASPP-CAS) ────
        lw_cfg = arch.get("lightweight_frontend", {})
        self.use_lightweight_frontend = lw_cfg.get("enabled", False)

        if self.use_lightweight_frontend:
            lw_out = lw_cfg.get("out_features", d_model)
            self.lightweight_frontend = EEGGhostDWASPPCASFrontEnd(
                in_freq_bins=self._cwt_n_freqs,
                hidden_channels=lw_cfg.get("hidden_channels", 64),
                out_features=lw_out,
                ghost_ratio=lw_cfg.get("ghost_ratio", 2),
                dwaspp_dilations=lw_cfg.get("dwaspp_dilations", [1, 2, 4, 8]),
                dropout=lw_cfg.get("dropout", 0.1),
                norm=lw_cfg.get("norm", "batchnorm"),
                activation=lw_cfg.get("activation", "gelu"),
                use_ghost=lw_cfg.get("use_ghost", True),
                use_dwaspp=lw_cfg.get("use_dwaspp", True),
                use_cas=lw_cfg.get("use_cas", True),
                use_normal_conv=lw_cfg.get("use_normal_conv", False),
            )
            # In the lightweight path, graph encoder input dim = lw_out (D)
            graph_d_in = lw_out
        else:
            self.lightweight_frontend = None
            graph_d_in = frontend_out_dim

        # ──── 2. Spatial Encoder ────
        spatial_cfg = arch.get("spatial", {})
        spatial_type = spatial_cfg.get("type", "none")

        if spatial_type == "graph":
            self.spatial_encoder = GraphSpatialEncoder(
                d_in=graph_d_in,
                d_model=d_model,
                n_layers=spatial_cfg.get("n_layers", 2),
                n_heads=spatial_cfg.get("n_heads", 4),
                dropout=spatial_cfg.get("dropout", 0.1),
                adjacency_type=spatial_cfg.get("adjacency", "distance"),
            )
        elif spatial_type == "region_pooling":
            self.spatial_encoder = nn.Sequential(
                nn.Linear(graph_d_in, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
        else:
            self.spatial_encoder = nn.Linear(graph_d_in, d_model)

        self.spatial_type = spatial_type

        # Node attention pool — used in lightweight path to collapse N → 1
        if self.use_lightweight_frontend:
            self.node_pool = NodeAttentionPool(d_model)
        else:
            self.node_pool = None

        # ──── 3. Conformer Encoder ────
        enc_cfg = arch.get("encoder", {})
        self.encoder = ConformerEncoder(
            d_model=d_model,
            n_layers=enc_cfg.get("n_layers", 4),
            n_heads=enc_cfg.get("n_heads", 8),
            d_ff=enc_cfg.get("d_ff", 512),
            conv_kernel_size=enc_cfg.get("conv_kernel_size", 31),
            dropout=enc_cfg.get("dropout", 0.1),
            attention_dropout=enc_cfg.get("attention_dropout", 0.1),
            use_relative_pos=enc_cfg.get("relative_pos_encoding", True),
            macaron=enc_cfg.get("macaron", True),
        )

        # ──── 4. Mamba (Optional) ────
        mamba_cfg = arch.get("mamba", {})
        self.use_mamba = mamba_cfg.get("enabled", False)
        if self.use_mamba:
            self.mamba = MambaModule(
                d_model=d_model,
                n_layers=mamba_cfg.get("n_layers", 1),
                d_state=mamba_cfg.get("d_state", 16),
                d_conv=mamba_cfg.get("d_conv", 4),
                expand=mamba_cfg.get("expand", 2),
                dropout=mamba_cfg.get("dropout", 0.1),
            )
        else:
            self.mamba = None

        # ──── 5. Output Heads ────
        heads_cfg = arch.get("heads", {})

        # Classification head(s)
        self.classification_heads = nn.ModuleDict()
        if "classification" in heads_cfg:
            cls_cfg = heads_cfg["classification"]
            if isinstance(n_classes, dict):
                for task_id, nc in n_classes.items():
                    self.classification_heads[task_id] = ClassificationHead(
                        d_model=d_model,
                        n_classes=nc,
                        dropout=cls_cfg.get("dropout", 0.3),
                    )
            else:
                self.classification_heads["default"] = ClassificationHead(
                    d_model=d_model,
                    n_classes=n_classes,
                    dropout=cls_cfg.get("dropout", 0.3),
                )

        # Retrieval head
        self.retrieval_head = None
        if "retrieval" in heads_cfg:
            ret_cfg = heads_cfg["retrieval"]
            self.retrieval_head = RetrievalHead(
                d_model=d_model,
                d_embed=ret_cfg.get("d_embed", 256),
                temperature=ret_cfg.get("temperature", 0.07),
            )

        # Generation head
        self.generation_head = None
        if "generation" in heads_cfg and vocab_size is not None:
            dec_cfg = arch.get("decoder", {})
            decoder = TransformerDecoder(
                vocab_size=vocab_size,
                d_model=d_model,
                n_layers=dec_cfg.get("n_layers", 4),
                n_heads=dec_cfg.get("n_heads", 8),
                d_ff=dec_cfg.get("d_ff", 1024),
                dropout=dec_cfg.get("dropout", 0.1),
                max_seq_len=dec_cfg.get("max_seq_len", 50),
                tie_embeddings=heads_cfg["generation"].get("tied_embeddings", True),
                ctc_weight=dec_cfg.get("ctc_weight", 0.3),
            )
            self.generation_head = GenerationHead(decoder=decoder, d_model=d_model)

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"NeuroGraphConformer initialized: "
            f"{total_params / 1e6:.2f}M total params, "
            f"{trainable_params / 1e6:.2f}M trainable"
        )

    def encode(
        self,
        x: torch.Tensor,
        adj: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode EEG input to latent representation.

        Parameters
        ----------
        x : torch.Tensor
            Raw EEG input, shape (batch, N, T).
        adj : torch.Tensor, optional
            Adjacency matrix for graph encoder, shape (N, N).
        mask : torch.Tensor, optional
            Padding mask.

        Returns
        -------
        torch.Tensor
            Encoder output, shape (batch, L, d_model).
        """
        if self.use_lightweight_frontend and self.lightweight_frontend is not None:
            # ---- Lightweight path: preserves electrodes + time ----
            # 1. Extract 4D time-frequency tensor: (B, N, F, T')
            x_tf = self.frontend.extract_time_frequency(x)
            B, N, F_dim, T_dim = x_tf.shape

            # 2. Electrode-wise local feature extraction: (B, N, D, T')
            x_local = self.lightweight_frontend(x_tf)
            D = x_local.shape[2]

            # 3. Rearrange for per-timestep graph encoding:
            #    (B, N, D, T') → (B, T', N, D) → (B*T', N, D)
            x_graph_in = x_local.permute(0, 3, 1, 2).contiguous()  # (B, T', N, D)
            x_graph_in = x_graph_in.reshape(B * T_dim, N, D)

            # 4. Graph spatial encoder at each time step: (B*T', N, D_g)
            if self.spatial_type == "graph" and adj is not None:
                x_graph = self.spatial_encoder(x_graph_in, adj)
            else:
                x_graph = self.spatial_encoder(x_graph_in)

            D_g = x_graph.shape[-1]

            # 5. Reshape back: (B*T', N, D_g) → (B, T', N, D_g)
            x_graph = x_graph.reshape(B, T_dim, N, D_g)

            # 6. Node attention pool: (B, T', N, D_g) → (B, T', D_g)
            x_seq = self.node_pool(x_graph)

        else:
            # ---- Original path: unchanged ----
            # Front-end: (batch, N, T) → (batch, N, d_frontend)
            x = self.frontend(x)

            # Spatial encoding
            if self.spatial_type == "graph" and adj is not None:
                x_seq = self.spatial_encoder(x, adj)
            else:
                x_seq = self.spatial_encoder(x)

        # Conformer expects (batch, L, d_model)
        h = self.encoder(x_seq, mask=mask)

        # Optional Mamba refinement
        if self.use_mamba and self.mamba is not None:
            h = self.mamba(h)

        return h

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        task: str = "classification",
        task_id: str = "default",
        tgt_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the full model.

        Parameters
        ----------
        x : torch.Tensor
            Raw EEG input, shape (batch, C, T).
        adj : torch.Tensor, optional
            Graph adjacency matrix.
        mask : torch.Tensor, optional
            Padding mask.
        task : str
            Active task: 'classification', 'retrieval', 'generation', 'all'.
        task_id : str
            Dataset/task-specific head ID (e.g., 'T1', 'T2', 'default').
        tgt_tokens : torch.Tensor, optional
            Target tokens for generation (teacher forcing).

        Returns
        -------
        dict[str, torch.Tensor]
            Output dictionary with keys depending on task.
        """
        # Encode
        h = self.encode(x, adj=adj, mask=mask)
        outputs = {"encoder_output": h}

        # Classification
        if task in ("classification", "all") and task_id in self.classification_heads:
            outputs["cls_logits"] = self.classification_heads[task_id](h, mask=mask)

        # Retrieval
        if task in ("retrieval", "all") and self.retrieval_head is not None:
            outputs["retrieval_emb"] = self.retrieval_head(h, mask=mask)

        # Generation
        if task in ("generation", "all") and self.generation_head is not None:
            if tgt_tokens is not None:
                gen_out = self.generation_head(h, tgt_tokens, encoder_mask=mask)
                outputs.update(gen_out)

        return outputs

    def get_param_groups(self, lr: float, weight_decay: float) -> list[dict]:
        """Get parameter groups with differential learning rates.

        Encoder uses lower LR; heads use higher LR.
        """
        encoder_params = []
        head_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "classification_heads" in name or "retrieval_head" in name or "generation_head" in name:
                head_params.append(param)
            else:
                encoder_params.append(param)

        return [
            {"params": encoder_params, "lr": lr, "weight_decay": weight_decay},
            {"params": head_params, "lr": lr * 10, "weight_decay": weight_decay},
        ]
