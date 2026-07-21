from __future__ import annotations
"""Stage 8.1 — QK^T reference.

Attention scores are a scaled matmul:

    scores[i, j] = dot(Q[i], K[j]) / sqrt(Dh)

This is the first attention-specific matmul. It differs from normal A@B because K is used with a logical
transpose:

    Q[Tq, Dh] @ K[Tk, Dh].T -> scores[Tq, Tk]

No causal mask or softmax yet. Keep this file simple.
"""

import math
import numpy as np


def qk_reference(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    if q.ndim != 2 or k.ndim != 2 or q.shape[1] != k.shape[1]:
        raise ValueError('Q and K must be [T,D] with same D')
    return q.astype(np.float32) @ k.astype(np.float32).T / math.sqrt(q.shape[1])


def qk_tiled_reference(q: np.ndarray, k: np.ndarray, block_m: int = 16, block_n: int = 16, block_d: int = 32) -> np.ndarray:
    tq, d = q.shape
    tk, d2 = k.shape
    if d != d2:
        raise ValueError('shape mismatch')
    scores = np.zeros((tq, tk), dtype=np.float32)
    scale = 1.0 / math.sqrt(d)
    for m0 in range(0, tq, block_m):
        for n0 in range(0, tk, block_n):
            m1 = min(m0 + block_m, tq)
            n1 = min(n0 + block_n, tk)
            acc = np.zeros((m1 - m0, n1 - n0), dtype=np.float32)
            for d0 in range(0, d, block_d):
                d1 = min(d0 + block_d, d)
                acc += q[m0:m1, d0:d1].astype(np.float32) @ k[n0:n1, d0:d1].astype(np.float32).T
            scores[m0:m1, n0:n1] = acc * scale
    return scores


def smoke_test() -> None:
    rng = np.random.default_rng(0)
    q = rng.normal(size=(13, 17)).astype(np.float32)
    k = rng.normal(size=(11, 17)).astype(np.float32)
    assert np.allclose(qk_tiled_reference(q, k, 5, 4, 7), qk_reference(q, k), atol=1e-5)
