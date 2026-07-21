"""Block reductions: sum and max.

Triton reductions usually happen within a block vector using tl.sum or tl.max.
"""
from __future__ import annotations
import numpy as np

def block_sum(x: np.ndarray, block_size: int=1024) -> float:
    partial=[]
    for pid in range((x.size+block_size-1)//block_size):
        offs=pid*block_size+np.arange(block_size); mask=offs<x.size
        partial.append(np.sum(np.where(mask, x[np.minimum(offs,x.size-1)], 0.0)))
    return float(np.sum(partial))

def block_max(x: np.ndarray, block_size: int=1024) -> float:
    partial=[]
    for pid in range((x.size+block_size-1)//block_size):
        offs=pid*block_size+np.arange(block_size); mask=offs<x.size
        vals=np.where(mask, x[np.minimum(offs,x.size-1)], -np.inf)
        partial.append(np.max(vals))
    return float(np.max(partial))

def smoke_test() -> None:
    x=np.random.default_rng(0).normal(size=123).astype(np.float32)
    assert abs(block_sum(x,16)-float(np.sum(x)))<1e-5
    assert block_max(x,16)==float(np.max(x))
