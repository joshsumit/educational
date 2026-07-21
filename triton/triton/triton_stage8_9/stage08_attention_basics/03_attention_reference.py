from __future__ import annotations
"""Stage 8.3 — Full scaled dot-product attention reference.

Formula:

    S = QK^T / sqrt(Dh)
    if causal:
        S = causal_mask(S)
    P = softmax(S)
    O = P V

This file is the correctness oracle for later Triton attention kernels.
"""

import math
import numpy as np


def softmax_stable(scores: np.ndarray) -> np.ndarray:
    m = np.max(scores, axis=-1, keepdims=True)
    e = np.exp(scores - m)
    e = np.where(np.isfinite(scores), e, 0.0)
    return e / np.sum(e, axis=-1, keepdims=True)


def attention_reference(q: np.ndarray, k: np.ndarray, v: np.ndarray, causal: bool = False) -> np.ndarray:
    if q.shape[1] != k.shape[1] or k.shape[0] != v.shape[0]:
        raise ValueError('shape mismatch')
    scores = q.astype(np.float32) @ k.astype(np.float32).T / math.sqrt(q.shape[1])
    if causal:
        rows = np.arange(scores.shape[0])[:, None]
        cols = np.arange(scores.shape[1])[None, :]
        scores = np.where(cols <= rows, scores, -np.inf)
    probs = softmax_stable(scores)
    return probs @ v.astype(np.float32)


def batched_attention_reference(q: np.ndarray, k: np.ndarray, v: np.ndarray, causal: bool = False) -> np.ndarray:
    """Reference for [B,T,D] single-head batches."""
    if q.ndim != 3:
        raise ValueError('expected [B,T,D]')
    out = []
    for b in range(q.shape[0]):
        out.append(attention_reference(q[b], k[b], v[b], causal=causal))
    return np.stack(out, axis=0)


def smoke_test() -> None:
    rng = np.random.default_rng(1)
    q = rng.normal(size=(7, 5)).astype(np.float32)
    k = rng.normal(size=(7, 5)).astype(np.float32)
    v = rng.normal(size=(7, 3)).astype(np.float32)
    out = attention_reference(q, k, v, causal=True)
    assert out.shape == (7, 3)
    qb = rng.normal(size=(2, 7, 5)).astype(np.float32)
    kb = rng.normal(size=(2, 7, 5)).astype(np.float32)
    vb = rng.normal(size=(2, 7, 3)).astype(np.float32)
    assert batched_attention_reference(qb, kb, vb, causal=False).shape == (2, 7, 3)
