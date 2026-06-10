"""
Brain-region pooling — map variable electrode montages to 9 canonical regions.

Defines the 10-20 region mapping and implements average/attention pooling.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ──── Canonical 9-region mapping ────
# Maps 10-20 electrode names to brain regions

REGION_MAP = {
    "LF": {  # Left Frontal
        "electrodes": [
            "Fp1", "AF3", "AF7", "F1", "F3", "F5", "F7", "FC1", "FC3", "FC5",
        ],
        "function": "Speech planning, Broca's area (inferior frontal)",
    },
    "RF": {  # Right Frontal
        "electrodes": [
            "Fp2", "AF4", "AF8", "F2", "F4", "F6", "F8", "FC2", "FC4", "FC6",
        ],
        "function": "Prosody, right-hemisphere language",
    },
    "LC": {  # Left Central
        "electrodes": ["C1", "C3", "C5", "CP1", "CP3", "CP5"],
        "function": "Motor cortex (articulatory motor imagery)",
    },
    "MC": {  # Midline Central
        "electrodes": ["Fz", "FCz", "Cz", "CPz"],
        "function": "Supplementary motor area",
    },
    "RC": {  # Right Central
        "electrodes": ["C2", "C4", "C6", "CP2", "CP4", "CP6"],
        "function": "Contralateral motor",
    },
    "LT": {  # Left Temporal
        "electrodes": ["T7", "TP7", "FT7", "FT9", "TP9"],
        "function": "Wernicke's area, auditory cortex",
    },
    "RT": {  # Right Temporal
        "electrodes": ["T8", "TP8", "FT8", "FT10", "TP10"],
        "function": "Auditory cortex",
    },
    "PA": {  # Parietal
        "electrodes": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "Pz", "POz"],
        "function": "Sensory integration",
    },
    "OC": {  # Occipital
        "electrodes": ["O1", "O2", "Oz", "PO3", "PO4", "PO7", "PO8"],
        "function": "Visual cortex",
    },
}

# Emotiv EPOC approximate mapping (14 channels)
EMOTIV_REGION_MAP = {
    "LF": ["AF3", "F3", "F7", "FC5"],
    "RF": ["AF4", "F4", "F8", "FC6"],
    "LT": ["T7"],
    "RT": ["T8"],
    "PA": ["P7", "P8"],
    "OC": ["O1", "O2"],
    # Note: LC, MC, RC are EMPTY for Emotiv EPOC
}

REGION_NAMES = list(REGION_MAP.keys())
N_REGIONS = len(REGION_NAMES)


def build_channel_to_region_mapping(
    ch_names: list[str],
    montage: str = "standard_1020",
) -> dict[str, list[int]]:
    """Build mapping from region names to channel indices.

    Parameters
    ----------
    ch_names : list[str]
        List of channel names in the dataset.
    montage : str
        Electrode montage type.

    Returns
    -------
    dict[str, list[int]]
        Mapping from region name to list of channel indices.
    """
    if montage == "emotiv_epoc":
        region_electrodes = EMOTIV_REGION_MAP
    else:
        region_electrodes = {r: info["electrodes"] for r, info in REGION_MAP.items()}

    # Normalize channel names for matching
    ch_names_upper = [ch.upper() for ch in ch_names]

    mapping = {}
    for region, electrodes in region_electrodes.items():
        indices = []
        for elec in electrodes:
            elec_upper = elec.upper()
            if elec_upper in ch_names_upper:
                idx = ch_names_upper.index(elec_upper)
                indices.append(idx)
        mapping[region] = indices

    return mapping


class RegionPooling(nn.Module):
    """Pool channels into brain regions.

    Parameters
    ----------
    ch_names : list[str]
        Channel names in the dataset.
    montage : str
        Electrode montage type.
    method : str
        Pooling method: 'mean' or 'attention'.
    d_in : int
        Input feature dimension per channel (for attention pooling).
    """

    def __init__(
        self,
        ch_names: list[str],
        montage: str = "standard_1020",
        method: str = "mean",
        d_in: int = 0,
    ):
        super().__init__()
        self.method = method
        self.n_regions = N_REGIONS
        self.region_names = REGION_NAMES

        # Build mapping
        self.mapping = build_channel_to_region_mapping(ch_names, montage)

        # Store as buffer for device transfer
        # Create index tensor: for each region, list of channel indices
        max_ch_per_region = max(len(v) for v in self.mapping.values()) if self.mapping else 1
        index_tensor = torch.full((N_REGIONS, max_ch_per_region), -1, dtype=torch.long)
        count_tensor = torch.zeros(N_REGIONS, dtype=torch.long)

        for i, region in enumerate(REGION_NAMES):
            indices = self.mapping.get(region, [])
            count_tensor[i] = len(indices)
            for j, idx in enumerate(indices):
                index_tensor[i, j] = idx

        self.register_buffer("region_indices", index_tensor)
        self.register_buffer("region_counts", count_tensor)

        # Attention weights (if method='attention')
        if method == "attention" and d_in > 0:
            self.attn_proj = nn.Linear(d_in, 1)
        else:
            self.attn_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool channels into regions.

        Parameters
        ----------
        x : torch.Tensor
            Per-channel features, shape (batch, C, d_feature).

        Returns
        -------
        torch.Tensor
            Region-pooled features, shape (batch, N_REGIONS, d_feature).
        """
        batch, n_ch, d = x.shape
        device = x.device

        outputs = torch.zeros(batch, self.n_regions, d, device=device)

        for i in range(self.n_regions):
            count = self.region_counts[i].item()
            if count == 0:
                # Region has no electrodes (e.g., Emotiv missing central)
                # Leave as zeros
                continue

            indices = self.region_indices[i, :count]  # (count,)
            region_data = x[:, indices, :]  # (batch, count, d)

            if self.method == "mean":
                outputs[:, i, :] = region_data.mean(dim=1)
            elif self.method == "attention" and self.attn_proj is not None:
                # Attention-weighted pooling
                attn_scores = self.attn_proj(region_data).squeeze(-1)  # (batch, count)
                attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch, count)
                outputs[:, i, :] = (attn_weights.unsqueeze(-1) * region_data).sum(dim=1)
            else:
                outputs[:, i, :] = region_data.mean(dim=1)

        return outputs
