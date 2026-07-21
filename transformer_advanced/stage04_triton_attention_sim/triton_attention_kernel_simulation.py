"""
Triton-style attention kernel simulation.

Triton kernels are often written as block programs:
    program_id(0) selects a query block
    program_id(1) selects batch/head
    tl.load loads Q/K/V blocks with masks
    tl.dot computes block scores
    online softmax updates row states

This file implements that structure in NumPy.
"""
from __future__ import annotations
import math
import numpy as np


def triton_style_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, block_m: int = 16, block_n: int = 32, causal: bool = False) -> np.ndarray:
    """
    Args:
        q, k, v: [T,D]
    Returns:
        out: [T,D]
    """
    t, d = q.shape
    out = np.zeros_like(q, dtype=np.float32)
    scale = 1.0 / math.sqrt(d)

    for pid_m, m0 in enumerate(range(0, t, block_m)):
        q_blk = q[m0:m0+block_m].astype(np.float32)
        rows = q_blk.shape[0]
        m_i = np.full((rows,), -np.inf, dtype=np.float32)
        l_i = np.zeros((rows,), dtype=np.float32)
        acc = np.zeros((rows, d), dtype=np.float32)
        for n0 in range(0, t, block_n):
            k_blk = k[n0:n0+block_n].astype(np.float32)
            v_blk = v[n0:n0+block_n].astype(np.float32)
            scores = q_blk @ k_blk.T * scale
            if causal:
                q_pos = np.arange(m0, m0 + rows)[:, None]
                k_pos = np.arange(n0, n0 + k_blk.shape[0])[None, :]
                scores = np.where(k_pos <= q_pos, scores, -np.inf)
            block_max = np.max(scores, axis=1)
            m_new = np.maximum(m_i, block_max)
            old_scale = np.exp(m_i - m_new)
            p = np.exp(scores - m_new[:, None])
            l_new = l_i * old_scale + np.sum(p, axis=1)
            acc = acc * old_scale[:, None] + p @ v_blk
            m_i, l_i = m_new, l_new
        out[m0:m0+rows] = acc / l_i[:, None]
    return out


def direct_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, causal: bool = False) -> np.ndarray:
    scores = q.astype(np.float32) @ k.astype(np.float32).T / math.sqrt(q.shape[1])
    if causal:
        mask = np.tril(np.ones(scores.shape, dtype=bool))
        scores = np.where(mask, scores, -np.inf)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    p = np.exp(scores)
    p = p / np.sum(p, axis=1, keepdims=True)
    return p @ v.astype(np.float32)


def smoke_test() -> None:
    rng = np.random.default_rng(4)
    q = rng.normal(size=(33,16)).astype(np.float16)
    k = rng.normal(size=(33,16)).astype(np.float16)
    v = rng.normal(size=(33,16)).astype(np.float16)
    got = triton_style_attention(q,k,v,8,11,causal=True)
    ref = direct_attention(q,k,v,causal=True)
    assert np.allclose(got, ref, atol=2e-3)
