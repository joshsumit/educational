"""Fused affine + ReLU.

Instead of writing intermediate y = x * scale + bias, a fused kernel computes and stores final output.
This reduces memory traffic.
"""
from __future__ import annotations
import numpy as np

def affine_relu_reference(x: np.ndarray, scale: float, bias: float) -> np.ndarray:
    return np.maximum(x*scale+bias, 0)

def affine_relu_blocked(x: np.ndarray, scale: float, bias: float, block_size: int=256) -> np.ndarray:
    out=np.empty_like(x)
    for pid in range((x.size+block_size-1)//block_size):
        offs=pid*block_size+np.arange(block_size); mask=offs<x.size
        vals=x[offs[mask]]*scale+bias
        out[offs[mask]]=np.maximum(vals,0)
    return out

def smoke_test() -> None:
    x=np.linspace(-2,2,31,dtype=np.float32)
    assert np.allclose(affine_relu_blocked(x,2,0.5,7), affine_relu_reference(x,2,0.5))
