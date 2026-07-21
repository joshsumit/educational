from __future__ import annotations
"""Stage 6.2 — Tiled/block matmul reference.

Triton matmul is tile-oriented. A program usually computes one C tile:

    C_tile[BLOCK_M, BLOCK_N]

It loops over K in chunks:

    for k0 in range(0, K, BLOCK_K):
        A_tile = A[m_offsets, k0 + k_offsets]
        B_tile = B[k0 + k_offsets, n_offsets]
        acc += A_tile @ B_tile

This CPU reference mirrors that structure. It is intentionally close to the real Triton kernel.
"""

import numpy as np


def matmul_tiled_reference(
    a: np.ndarray,
    b: np.ndarray,
    block_m: int = 16,
    block_n: int = 16,
    block_k: int = 32,
) -> np.ndarray:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError('rank-2 inputs expected')
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError('shape mismatch')
    c = np.zeros((m, n), dtype=np.float32)
    for m0 in range(0, m, block_m):
        for n0 in range(0, n, block_n):
            m1 = min(m0 + block_m, m)
            n1 = min(n0 + block_n, n)
            acc = np.zeros((m1 - m0, n1 - n0), dtype=np.float32)
            for k0 in range(0, k, block_k):
                k1 = min(k0 + block_k, k)
                acc += a[m0:m1, k0:k1].astype(np.float32) @ b[k0:k1, n0:n1].astype(np.float32)
            c[m0:m1, n0:n1] = acc
    return c


def estimate_tiled_work(m: int, n: int, k: int, block_m: int, block_n: int, block_k: int) -> dict[str, int]:
    num_m = (m + block_m - 1) // block_m
    num_n = (n + block_n - 1) // block_n
    num_k = (k + block_k - 1) // block_k
    return {
        'num_output_tiles': num_m * num_n,
        'k_tiles_per_output_tile': num_k,
        'program_like_iterations': num_m * num_n * num_k,
    }


def smoke_test() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(size=(17, 19)).astype(np.float32)
    b = rng.normal(size=(19, 11)).astype(np.float32)
    out = matmul_tiled_reference(a, b, block_m=5, block_n=4, block_k=7)
    assert np.allclose(out, a @ b, atol=1e-5)
    model = estimate_tiled_work(17, 11, 19, 5, 4, 7)
    assert model['num_output_tiles'] == 12
    assert model['k_tiles_per_output_tile'] == 3
