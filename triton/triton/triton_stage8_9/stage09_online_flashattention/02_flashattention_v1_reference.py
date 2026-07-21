from __future__ import annotations
"""Stage 9.2 — FlashAttention v1-style reference.

This is a CPU reference of the central FlashAttention idea:

    Do not materialize the full score matrix S[T,T].
    Do not materialize the full probability matrix P[T,T].

Instead, for each query block and key/value block:

    1. compute scores block
    2. update online softmax stats m and l
    3. rescale old output accumulator
    4. add new probability-weighted V block

This reference is intentionally simple and focuses on correctness. It is not optimized CPU code.
"""

import math
import numpy as np


def attention_reference(q: np.ndarray, k: np.ndarray, v: np.ndarray, causal: bool = True) -> np.ndarray:
    scores = q.astype(np.float32) @ k.astype(np.float32).T / math.sqrt(q.shape[1])
    if causal:
        rows = np.arange(scores.shape[0])[:, None]
        cols = np.arange(scores.shape[1])[None, :]
        scores = np.where(cols <= rows, scores, -np.inf)
    m = np.max(scores, axis=1, keepdims=True)
    e = np.exp(scores - m)
    e = np.where(np.isfinite(scores), e, 0.0)
    p = e / np.sum(e, axis=1, keepdims=True)
    return p @ v.astype(np.float32)


def flashattention_v1_reference(q: np.ndarray, k: np.ndarray, v: np.ndarray, block_m: int = 16, block_n: int = 16, causal: bool = True) -> np.ndarray:
    tq, d = q.shape
    tk, d2 = k.shape
    if d != d2 or tk != v.shape[0]:
        raise ValueError('shape mismatch')
    dv = v.shape[1]
    scale = 1.0 / math.sqrt(d)
    out = np.zeros((tq, dv), dtype=np.float32)

    for m0 in range(0, tq, block_m):
        m1 = min(m0 + block_m, tq)
        q_block = q[m0:m1].astype(np.float32)
        rows = np.arange(m0, m1)[:, None]
        m_i = np.full((m1 - m0,), -np.inf, dtype=np.float32)
        l_i = np.zeros((m1 - m0,), dtype=np.float32)
        acc = np.zeros((m1 - m0, dv), dtype=np.float32)

        for n0 in range(0, tk, block_n):
            n1 = min(n0 + block_n, tk)
            cols = np.arange(n0, n1)[None, :]
            scores = q_block @ k[n0:n1].astype(np.float32).T * scale
            if causal:
                scores = np.where(cols <= rows, scores, -np.inf)

            m_block = np.max(scores, axis=1)
            m_new = np.maximum(m_i, m_block)
            old_scale = np.exp(m_i - m_new)
            old_scale = np.where(np.isfinite(m_i), old_scale, 0.0)
            p = np.exp(scores - m_new[:, None])
            p = np.where(np.isfinite(scores), p, 0.0)
            l_new = l_i * old_scale + np.sum(p, axis=1)
            acc = acc * old_scale[:, None] + p @ v[n0:n1].astype(np.float32)
            m_i = m_new
            l_i = l_new

        out[m0:m1] = acc / l_i[:, None]
    return out


def smoke_test() -> None:
    rng = np.random.default_rng(6)
    q = rng.normal(size=(23, 17)).astype(np.float32)
    k = rng.normal(size=(23, 17)).astype(np.float32)
    v = rng.normal(size=(23, 11)).astype(np.float32)
    out = flashattention_v1_reference(q, k, v, block_m=5, block_n=7, causal=True)
    exp = attention_reference(q, k, v, causal=True)
    assert np.allclose(out, exp, atol=1e-5)
