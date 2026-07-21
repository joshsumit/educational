"""Masked load/store references.

Triton boundary handling pattern:
    x = tl.load(ptr + offsets, mask=offsets < n, other=0.0)
    tl.store(out + offsets, y, mask=offsets < n)
"""
from __future__ import annotations
import numpy as np

def masked_load(x: np.ndarray, offsets: np.ndarray, mask: np.ndarray, other: float = 0.0) -> np.ndarray:
    y=np.full(offsets.shape, other, dtype=x.dtype)
    y[mask]=x[offsets[mask]]
    return y

def masked_store(out: np.ndarray, offsets: np.ndarray, values: np.ndarray, mask: np.ndarray) -> None:
    out[offsets[mask]]=values[mask]

def vector_add_masked(a: np.ndarray, b: np.ndarray, block_size: int) -> np.ndarray:
    n=a.size; out=np.empty_like(a)
    for pid in range((n+block_size-1)//block_size):
        offs=pid*block_size+np.arange(block_size)
        mask=offs<n
        av=masked_load(a,offs,mask,0); bv=masked_load(b,offs,mask,0)
        masked_store(out,offs,av+bv,mask)
    return out

def smoke_test() -> None:
    a=np.arange(10,dtype=np.float32); b=2*a
    assert np.allclose(vector_add_masked(a,b,4), a+b)
