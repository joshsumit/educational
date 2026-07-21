"""Softmax and LayerNorm references in Triton-friendly row-block style."""
from __future__ import annotations
import numpy as np

def row_softmax(x: np.ndarray) -> np.ndarray:
    m=np.max(x,axis=1,keepdims=True); e=np.exp(x-m); return e/np.sum(e,axis=1,keepdims=True)

def row_softmax_blocked(x: np.ndarray) -> np.ndarray:
    out=np.empty_like(x,dtype=np.float32)
    for r in range(x.shape[0]):
        row=x[r].astype(np.float32); row=row-np.max(row); e=np.exp(row); out[r]=e/np.sum(e)
    return out

def layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float=1e-5) -> np.ndarray:
    mean=np.mean(x,axis=1,keepdims=True); var=np.mean((x-mean)**2,axis=1,keepdims=True)
    return (x-mean)/np.sqrt(var+eps)*gamma+beta

def smoke_test() -> None:
    rng=np.random.default_rng(1); x=rng.normal(size=(5,13)).astype(np.float32)
    assert np.allclose(row_softmax_blocked(x), row_softmax(x), atol=1e-6)
    g=np.ones(13,dtype=np.float32); b=np.zeros(13,dtype=np.float32)
    y=layernorm(x,g,b); assert y.shape==x.shape and np.allclose(np.mean(y,axis=1),0,atol=1e-5)
