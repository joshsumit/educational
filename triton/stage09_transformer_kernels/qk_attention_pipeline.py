"""Transformer attention kernels after Triton basics.

This file implements QK^T, causal masking, softmax, and PV using Triton-style blocks in NumPy.
"""
from __future__ import annotations
import math
import numpy as np

def qk_blocked(q: np.ndarray, k: np.ndarray, block_m:int=16, block_n:int=16) -> np.ndarray:
    t,d=q.shape; scores=np.empty((t,t),dtype=np.float32)
    for m0 in range(0,t,block_m):
        for n0 in range(0,t,block_n):
            scores[m0:m0+block_m,n0:n0+block_n]=q[m0:m0+block_m].astype(np.float32) @ k[n0:n0+block_n].astype(np.float32).T / math.sqrt(d)
    return scores

def attention_blocked(q: np.ndarray, k: np.ndarray, v: np.ndarray, causal:bool=False) -> np.ndarray:
    s=qk_blocked(q,k)
    if causal:
        s=np.where(np.tril(np.ones_like(s,dtype=bool)),s,-np.inf)
    p=np.exp(s-np.max(s,axis=1,keepdims=True)); p=p/np.sum(p,axis=1,keepdims=True)
    return p @ v.astype(np.float32)

def smoke_test() -> None:
    rng=np.random.default_rng(4); q=rng.normal(size=(13,8)).astype(np.float32); k=rng.normal(size=(13,8)).astype(np.float32); v=rng.normal(size=(13,8)).astype(np.float32)
    s=q@k.T/math.sqrt(8); mask=np.tril(np.ones_like(s,dtype=bool)); p=np.exp(np.where(mask,s,-np.inf)-np.max(np.where(mask,s,-np.inf),axis=1,keepdims=True)); p=p/np.sum(p,axis=1,keepdims=True)
    assert np.allclose(attention_blocked(q,k,v,True), p@v, atol=1e-5)
