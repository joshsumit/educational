"""Bias + GELU fusion.

Common Transformer fusion:
    y = GELU(x + bias)
A fused kernel avoids storing x+bias to global memory.
"""
from __future__ import annotations
import numpy as np

def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5*x*(1.0+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3)))

def fused_bias_gelu(x: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return gelu(x+bias)

def fused_bias_gelu_blocked(x: np.ndarray, bias: np.ndarray, block_size:int=256) -> np.ndarray:
    flat=x.reshape(-1); out=np.empty_like(flat,dtype=np.float32)
    cols=x.shape[-1]
    for pid in range((flat.size+block_size-1)//block_size):
        offs=pid*block_size+np.arange(block_size); mask=offs<flat.size
        col=offs[mask] % cols
        out[offs[mask]]=gelu(flat[offs[mask]]+bias[col])
    return out.reshape(x.shape)

def smoke_test() -> None:
    rng=np.random.default_rng(3); x=rng.normal(size=(4,9)).astype(np.float32); b=rng.normal(size=(9,)).astype(np.float32)
    assert np.allclose(fused_bias_gelu_blocked(x,b,7), fused_bias_gelu(x,b), atol=1e-6)
