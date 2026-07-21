from __future__ import annotations
"""Stage 7.2 — Batched matmul reference.

Batched matmul convention:

    A[B, M, K] @ Bmat[B, K, N] = C[B, M, N]

Why this matters:
    - Multi-head attention has per-batch/per-head matmul patterns.
    - QK and PV are naturally batched across batch/head dimensions.
    - Later kernels can flatten batch and head into one batch dimension.
"""

import numpy as np


def batched_matmul_reference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError('rank-3 inputs expected')
    batch, m, k = a.shape
    batch2, k2, n = b.shape
    if batch != batch2 or k != k2:
        raise ValueError('shape mismatch')
    c = np.empty((batch, m, n), dtype=np.float32)
    for bb in range(batch):
        c[bb] = a[bb].astype(np.float32) @ b[bb].astype(np.float32)
    return c


def flatten_batch_head(x: np.ndarray) -> np.ndarray:
    """Flatten [B, H, T, D] into [B*H, T, D], common before batched QK/PV."""
    if x.ndim != 4:
        raise ValueError('rank-4 input expected')
    b, h, t, d = x.shape
    return x.reshape(b * h, t, d)


def smoke_test() -> None:
    rng = np.random.default_rng(7)
    a = rng.normal(size=(4, 5, 7)).astype(np.float32)
    b = rng.normal(size=(4, 7, 3)).astype(np.float32)
    out = batched_matmul_reference(a, b)
    assert np.allclose(out, np.matmul(a, b), atol=1e-5)
    x = rng.normal(size=(2, 3, 4, 5)).astype(np.float32)
    assert flatten_batch_head(x).shape == (6, 4, 5)
