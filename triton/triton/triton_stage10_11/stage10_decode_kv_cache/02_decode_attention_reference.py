from __future__ import annotations
"""Stage 10.2 — Single-query decode attention reference.

Decode attention for one head:

    q       [D]
    k_cache [S, D]
    v_cache [S, Dv]

    scores = k_cache @ q / sqrt(D)
    probs  = softmax(scores)
    out    = probs @ v_cache

This file includes both a direct reference and an online-softmax blocked reference. The blocked version mirrors
what a decode kernel does: stream K/V blocks and maintain running softmax state.
"""

import math
import numpy as np


def decode_attention_reference(q: np.ndarray, k_cache: np.ndarray, v_cache: np.ndarray) -> np.ndarray:
    scores = k_cache.astype(np.float32) @ q.astype(np.float32) / math.sqrt(q.size)
    p = np.exp(scores - np.max(scores))
    p = p / np.sum(p)
    return p @ v_cache.astype(np.float32)


def decode_attention_online_reference(q: np.ndarray, k_cache: np.ndarray, v_cache: np.ndarray, block_n: int = 128) -> np.ndarray:
    d = q.size
    dv = v_cache.shape[1]
    m = -math.inf
    l = 0.0
    acc = np.zeros((dv,), dtype=np.float32)
    for n0 in range(0, k_cache.shape[0], block_n):
        k = k_cache[n0:n0+block_n].astype(np.float32)
        v = v_cache[n0:n0+block_n].astype(np.float32)
        scores = k @ q.astype(np.float32) / math.sqrt(d)
        mb = float(np.max(scores))
        m_new = max(m, mb)
        old_scale = 0.0 if m == -math.inf else math.exp(m - m_new)
        p = np.exp(scores - m_new)
        l = l * old_scale + float(np.sum(p))
        acc = acc * old_scale + p @ v
        m = m_new
    return acc / l


def smoke_test() -> None:
    rng = np.random.default_rng(0)
    q = rng.normal(size=(16,)).astype(np.float32)
    k = rng.normal(size=(65, 16)).astype(np.float32)
    v = rng.normal(size=(65, 12)).astype(np.float32)
    assert np.allclose(decode_attention_online_reference(q, k, v, 17), decode_attention_reference(q, k, v), atol=1e-5)
