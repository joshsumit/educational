"""Vector add: first Triton-style kernel pattern."""
from __future__ import annotations
import numpy as np

def vector_add_reference(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x+y

def vector_add_blocked(x: np.ndarray, y: np.ndarray, block_size: int = 1024) -> np.ndarray:
    out=np.empty_like(x)
    for pid in range((x.size+block_size-1)//block_size):
        offs=pid*block_size+np.arange(block_size)
        mask=offs<x.size
        out[offs[mask]]=x[offs[mask]]+y[offs[mask]]
    return out

def smoke_test() -> None:
    x=np.arange(100,dtype=np.float32); y=x*3
    assert np.allclose(vector_add_blocked(x,y,17), vector_add_reference(x,y))
