import torch
import pytest

from src.features.wavelet import FilterBankFrontEnd
from src.features.graph_builder import GraphBuilder

def test_feature_extraction_pipeline():
    """
    Verifies that the FilterBank + GraphBuilder pipeline successfully
    transforms a raw (B, C, T) EEG batch into the (x_node, adj) format
    required by the Conformer, resolving the adjacency matrix crash risk.
    """
    B = 2
    C = 61 # Harmonized channel layout
    T = 250 # 1 second at 250Hz
    
    # --- 1. Wavelet Front-End ---
    wavelet = FilterBankFrontEnd(in_channels=C, num_filters=16, kernel_size=64, stride=4)
    x_raw = torch.randn(B, C, T)
    
    x_tf = wavelet(x_raw)
    
    # Expected shape: (B, C, F, T') where F=16
    # T' = floor((T + 2*padding - kernel_size) / stride) + 1
    # padding = 64//2 = 32
    # T' = floor((250 + 64 - 64) / 4) + 1 = 63
    F_bins = 16
    T_prime = 63
    
    assert x_tf.shape == (B, C, F_bins, T_prime), f"Wavelet output shape mismatch: {x_tf.shape}"
    
    # --- 2. Graph Builder (Dynamic Adjacency) ---
    graph_dynamic = GraphBuilder(
        num_nodes=C, 
        freq_bins=F_bins, 
        time_steps=T_prime, 
        out_features=128, 
        dynamic_adj=True
    )
    
    x_node, adj = graph_dynamic(x_tf)
    
    # Check node feature shape
    assert x_node.shape == (B, C, 128), f"Node feature shape mismatch: {x_node.shape}"
    
    # Check dynamic adjacency shape
    assert adj.shape == (B, C, C), f"Dynamic adjacency shape mismatch: {adj.shape}"
    
    # Verify adjacency weights form valid probability distributions (sum to 1 over neighborhood)
    assert torch.allclose(adj.sum(dim=-1), torch.ones(B, C), atol=1e-6), "Adjacency matrix weights do not sum to 1"
    
    # --- 3. Graph Builder (Static Adjacency) ---
    graph_static = GraphBuilder(
        num_nodes=C, 
        freq_bins=F_bins, 
        time_steps=T_prime, 
        out_features=128, 
        dynamic_adj=False
    )
    
    x_node_static, adj_static = graph_static(x_tf)
    
    assert x_node_static.shape == (B, C, 128)
    assert adj_static.shape == (C, C), "Static adjacency should not have batch dimension"
    assert torch.allclose(adj_static.sum(dim=-1), torch.ones(C), atol=1e-6)
