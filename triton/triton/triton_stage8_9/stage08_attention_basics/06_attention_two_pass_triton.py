from __future__ import annotations
"""Stage 8.6 — Two-pass teaching attention.

This file intentionally uses the simple pipeline:

    1. scores = QK^T / sqrt(D)
    2. probs = causal_softmax(scores)
    3. out = probs @ V

Why this is useful:
    - Easy to debug.
    - Each step maps to a previous stage: matmul, softmax, matmul.
    - Good for learning correctness and shapes.

Why this is not production optimal:
    - It materializes the full [T,T] score matrix.
    - It materializes the full [T,T] probability matrix.
    - Memory grows as O(T^2).

FlashAttention exists to avoid these materialized matrices.
"""

import math
import numpy as np

try:
    import torch
except Exception:
    torch = None


def attention_two_pass_reference(q: np.ndarray, k: np.ndarray, v: np.ndarray, causal: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = q.astype(np.float32) @ k.astype(np.float32).T / math.sqrt(q.shape[1])
    if causal:
        rows = np.arange(scores.shape[0])[:, None]
        cols = np.arange(scores.shape[1])[None, :]
        scores = np.where(cols <= rows, scores, -np.inf)
    m = np.max(scores, axis=1, keepdims=True)
    e = np.exp(scores - m)
    e = np.where(np.isfinite(scores), e, 0.0)
    probs = e / np.sum(e, axis=1, keepdims=True)
    out = probs @ v.astype(np.float32)
    return scores, probs, out


def two_pass_memory_bytes(t: int, d: int, dv: int | None = None, bytes_per_element: int = 4) -> dict[str, int]:
    """First-order memory footprint for materialized attention."""
    dv = d if dv is None else dv
    qkv = (t * d + t * d + t * dv) * bytes_per_element
    scores = t * t * bytes_per_element
    probs = t * t * bytes_per_element
    out = t * dv * bytes_per_element
    return {'qkv_bytes': qkv, 'scores_bytes': scores, 'probs_bytes': probs, 'out_bytes': out, 'total_bytes': qkv + scores + probs + out}


def smoke_test() -> None:
    rng = np.random.default_rng(4)
    q = rng.normal(size=(6, 5)).astype(np.float32)
    k = rng.normal(size=(6, 5)).astype(np.float32)
    v = rng.normal(size=(6, 3)).astype(np.float32)
    s, p, o = attention_two_pass_reference(q, k, v, causal=True)
    assert s.shape == (6, 6)
    assert p.shape == (6, 6)
    assert o.shape == (6, 3)
    assert two_pass_memory_bytes(4, 8)['scores_bytes'] == 64

if __name__ == '__main__':
    smoke_test(); print('ok')
