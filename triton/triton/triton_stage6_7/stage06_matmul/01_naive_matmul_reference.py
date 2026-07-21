from __future__ import annotations
"""Stage 6.1 — Naive matmul reference.

This is the simple mathematical baseline:

    for m in M:
      for n in N:
        acc = 0
        for k in K:
          acc += A[m, k] * B[k, n]
        C[m, n] = acc

Why keep this file?
    - It is the correctness oracle.
    - It makes M/N/K semantics explicit.
    - It is the first thing to explain in interviews before discussing tiling.

Why it is slow:
    - Poor cache behavior for large matrices.
    - Python loops are slow.
    - No vectorization, no tiling, no tensor cores.
"""

import numpy as np


def matmul_naive(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError('rank-2 inputs expected')
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError('shape mismatch')
    c = np.zeros((m, n), dtype=np.float32)
    for i in range(m):
        for j in range(n):
            acc = 0.0
            for kk in range(k):
                acc += float(a[i, kk]) * float(b[kk, j])
            c[i, j] = acc
    return c


def matmul_numpy_reference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a.astype(np.float32) @ b.astype(np.float32)


def smoke_test() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(5, 7)).astype(np.float32)
    b = rng.normal(size=(7, 3)).astype(np.float32)
    assert np.allclose(matmul_naive(a, b), matmul_numpy_reference(a, b), atol=1e-5)
