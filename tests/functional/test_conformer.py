import torch
import pytest
from src.models.conformer import ConformerEncoder

def test_conformer_encoder_forward():
    batch_size = 2
    seq_len = 100
    d_model = 64
    
    # Create encoder
    encoder = ConformerEncoder(
        d_model=d_model,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        conv_kernel_size=15,
        dropout=0.1,
        attention_dropout=0.1,
    )
    
    # Input tensor
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    
    # Optional mask (True means valid, False means padding)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[1, 80:] = False  # Mask out the last 20 timesteps of second item
    
    out = encoder(x, mask=mask)
    
    # Check shape
    assert out.shape == (batch_size, seq_len, d_model)
    
    # Check gradient flow
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None
    
    # Test without mask
    out_nomask = encoder(x)
    assert out_nomask.shape == (batch_size, seq_len, d_model)
