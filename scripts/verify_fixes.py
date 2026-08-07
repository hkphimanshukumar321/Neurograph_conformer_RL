"""Quick smoke-test for all fixed modules."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from omegaconf import OmegaConf

print("=" * 60)
print("NeuroGraph-Conformer-RL Verification Script")
print("=" * 60)

# Test 1: Conformer with relative pos encoding
from src.models.conformer import ConformerEncoder
enc = ConformerEncoder(d_model=64, n_layers=2, n_heads=4, d_ff=128, conv_kernel_size=7, use_relative_pos=True)
x = torch.randn(2, 10, 64)
out = enc(x)
print(f"1. ConformerEncoder: {x.shape} -> {out.shape}  OK")

# Test 2: MHSA with relative position bias
from src.models.attention import MultiHeadSelfAttention
mhsa = MultiHeadSelfAttention(d_model=64, n_heads=4)
y = mhsa(x)
print(f"2. MHSA + rel_pos_bias: {x.shape} -> {y.shape}  OK")

# Test 3: Mamba fallback with proper SSM
from src.models.mamba_module import MambaModule
mamba = MambaModule(d_model=64, n_layers=1, d_state=8, expand=2)
m_out = mamba(x)
print(f"3. MambaModule SSM: {x.shape} -> {m_out.shape}  OK")

# Test 4: Full model instantiation
cfg = OmegaConf.load("configs/models/neurograph_full.yaml")
from src.models.neurograph import NeuroGraphConformer
model = NeuroGraphConformer(cfg.model, n_channels=9, n_samples=500, n_classes=10, vocab_size=100)
inp = torch.randn(2, 9, 500)
outputs = model(inp, task="all")
cls_shape = outputs["cls_logits"].shape
print(f"4. NeuroGraphConformer: cls_logits={cls_shape}  OK")
if "retrieval_emb" in outputs:
    print(f"   retrieval_emb={outputs['retrieval_emb'].shape}  OK")
if model.generation_head is not None:
    print(f"   generation_head=present  OK")
total = sum(p.numel() for p in model.parameters()) / 1e6
print(f"   Total params: {total:.2f}M")

# Test 5: InfoNCELoss
from src.models.losses import InfoNCELoss
loss_fn = InfoNCELoss(temperature=0.07)
a = torch.randn(4, 32)
b = torch.randn(4, 32)
a = torch.nn.functional.normalize(a, dim=-1)
b = torch.nn.functional.normalize(b, dim=-1)
loss = loss_fn(a, b)
print(f"5. InfoNCELoss: {loss.item():.4f}  OK")

# Test 6: Factory with tokenizer
from src.datasets.tokenizer import Tokenizer
tok = Tokenizer()
print(f"6. Tokenizer: vocab_size={tok.vocab_size}  OK")

print()
print("ALL VERIFICATIONS PASSED")
