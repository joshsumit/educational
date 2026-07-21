"""
`mma.sync` walkthrough using executable NumPy reference code.

Real CUDA instruction example:
    mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16

Meaning:
    - M tile height = 16
    - N tile width  = 8
    - K reduction   = 16
    - A layout row-major, B layout col-major in the instruction contract
    - input/output types are FP16 in this simplified example

This file does not issue real tensor-core instructions. It models the numerical effect and
fragment ownership clearly enough for interview explanations.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class MmaShape:
    m: int = 16
    n: int = 8
    k: int = 16


def mma_sync_reference(a_frag: np.ndarray, b_frag: np.ndarray, c_frag: np.ndarray) -> np.ndarray:
    """
    Numerical effect of one MMA instruction:
        D = A @ B + C

    Args:
        a_frag: [16,16]
        b_frag: [16,8]
        c_frag: [16,8]

    In real hardware these fragments are distributed across 32 lanes as registers.
    No single lane owns the full matrices.
    """
    if a_frag.shape != (16, 16) or b_frag.shape != (16, 8) or c_frag.shape != (16, 8):
        raise ValueError('expected A[16,16], B[16,8], C[16,8]')
    return a_frag.astype(np.float32) @ b_frag.astype(np.float32) + c_frag.astype(np.float32)


def warp_mma_tile_gemm(a: np.ndarray, b: np.ndarray, block_m: int = 16, block_n: int = 8, block_k: int = 16) -> np.ndarray:
    """
    GEMM using a warp-level MMA tile shape.

    Beginner:
        This looks like tiled matmul.

    Advanced:
        Each inner loop corresponds to one logical MMA instruction over a K tile.
        Production kernels run many such instructions per warp and keep accumulators in registers.
    """
    m, k_total = a.shape
    k2, n = b.shape
    if k_total != k2:
        raise ValueError('inner dimensions must match')
    out = np.zeros((m, n), dtype=np.float32)
    for m0 in range(0, m, block_m):
        for n0 in range(0, n, block_n):
            acc = np.zeros((min(block_m, m - m0), min(block_n, n - n0)), dtype=np.float32)
            for k0 in range(0, k_total, block_k):
                a_tile = a[m0:m0+acc.shape[0], k0:k0+block_k].astype(np.float32)
                b_tile = b[k0:k0+block_k, n0:n0+acc.shape[1]].astype(np.float32)
                acc += a_tile @ b_tile
            out[m0:m0+acc.shape[0], n0:n0+acc.shape[1]] = acc
    return out


def estimate_mma_instruction_count(m: int, n: int, k: int, shape: MmaShape = MmaShape()) -> int:
    """Number of logical MMA ops required for a matrix tile."""
    return ((m + shape.m - 1) // shape.m) * ((n + shape.n - 1) // shape.n) * ((k + shape.k - 1) // shape.k)


def smoke_test() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(16,16)).astype(np.float16)
    b = rng.normal(size=(16,8)).astype(np.float16)
    c = rng.normal(size=(16,8)).astype(np.float16)
    d = mma_sync_reference(a, b, c)
    assert np.allclose(d, a.astype(np.float32) @ b.astype(np.float32) + c.astype(np.float32), atol=1e-3)
    big_a = rng.normal(size=(34,48)).astype(np.float16)
    big_b = rng.normal(size=(48,17)).astype(np.float16)
    got = warp_mma_tile_gemm(big_a, big_b)
    assert np.allclose(got, big_a.astype(np.float32) @ big_b.astype(np.float32), atol=1e-2)
    assert estimate_mma_instruction_count(16,8,16) == 1
