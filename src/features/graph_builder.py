import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GraphBuilder(nn.Module):
    """
    Converts Time-Frequency features into Graph Nodes and Adjacency Matrices.
    Guarantees the output of `(x, adj)` required by the GraphSpatialEncoder,
    resolving the code-level risk of a missing adjacency matrix.
    """
    def __init__(self, num_nodes: int, freq_bins: int, time_steps: int, out_features: int, dynamic_adj: bool = True):
        super().__init__()
        self.dynamic_adj = dynamic_adj
        self.num_nodes = num_nodes
        
        # Flattened feature dimension (F * T')
        in_features = freq_bins * time_steps
        
        # Projection for node features
        self.feature_proj = nn.Linear(in_features, out_features)
        
        # If not fully dynamic, we learn a static adjacency matrix
        if not dynamic_adj:
            # Initialize with uniform connectivity
            self.static_adj = nn.Parameter(torch.ones(num_nodes, num_nodes) / num_nodes)
            
    def forward(self, x_tf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_tf: (B, C, F, T') time-frequency features from Wavelet Front-End
        Returns:
            x_node: (B, C, D) node features where C is the number of nodes
            adj: (B, C, C) or (C, C) adjacency matrix
        """
        B, C, F_bins, T_prime = x_tf.shape
        assert C == self.num_nodes, f"Expected {self.num_nodes} channels/nodes, got {C}"
        
        # Flatten F and T' to create the initial raw node representation
        # (B, C, F * T')
        x_flat = x_tf.view(B, C, F_bins * T_prime)
        
        # Project to D dimension: (B, C, D)
        x_node = self.feature_proj(x_flat)
        x_node = F.gelu(x_node)
        
        if self.dynamic_adj:
            # Compute dynamic correlation matrix batch-wise
            # (B, C, D) x (B, D, C) -> (B, C, C)
            x_norm = F.normalize(x_node, p=2, dim=-1)
            adj = torch.bmm(x_norm, x_norm.transpose(1, 2))
            
            # Softmax to guarantee valid adjacency weights
            adj = F.softmax(adj / math.sqrt(x_node.size(-1)), dim=-1)
        else:
            # Use static learnable adjacency
            adj = F.softmax(self.static_adj, dim=-1)
            
        return x_node, adj
