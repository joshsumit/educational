"""Decode attention as a one-query Triton-style kernel.

Decode is usually memory-bound: a single query streams the full K/V cache.
"""
from __future__ import annotations
import math
import numpy as np

def decode_attention(q: np.ndarray, k_cache: np.ndarray, v_cache: np.ndarray, block_n:int=128) -> np.ndarray:
    m=-np.inf; l=0.0; acc=np.zeros((v_cache.shape[1],),dtype=np.float32)
    for n0 in range(0,k_cache.shape[0],block_n):
        k=k_cache[n0:n0+block_n].astype(np.float32); v=v_cache[n0:n0+block_n].astype(np.float32)
        scores=k@q.astype(np.float32)/math.sqrt(q.size)
        mb=float(np.max(scores)); m_new=max(m,mb)
        old=math.exp(m-m_new) if m!=-np.inf else 0.0
        p=np.exp(scores-m_new)
        l=l*old+float(np.sum(p)); acc=acc*old+p@v; m=m_new
    return acc/l

def decode_memory_bytes(seq:int, head_dim:int, bytes_per_elem:int=2) -> int:
    return 2*seq*head_dim*bytes_per_elem

def smoke_test() -> None:
    rng=np.random.default_rng(5); q=rng.normal(size=(16,)).astype(np.float32); k=rng.normal(size=(65,16)).astype(np.float32); v=rng.normal(size=(65,16)).astype(np.float32)
    s=k@q/math.sqrt(16); p=np.exp(s-np.max(s)); p=p/np.sum(p)
    assert np.allclose(decode_attention(q,k,v,17), p@v, atol=1e-5)
    assert decode_memory_bytes(10,8,2)==320
