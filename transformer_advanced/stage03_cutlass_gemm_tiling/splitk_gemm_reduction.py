"""
Split-K GEMM reduction.

Problem:
    If M and N are small but K is huge, there may not be enough CTA parallelism.

Solution:
    Split the K dimension across multiple CTAs. Each CTA computes a partial C tile.
    Then reduce partial C tiles.

Tradeoff:
    More parallelism, but extra reduction memory traffic and synchronization.
"""
from __future__ import annotations
import numpy as np


def splitk_gemm(a: np.ndarray, b: np.ndarray, split_k: int = 2) -> tuple[np.ndarray, list[np.ndarray]]:
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError('inner dimensions must match')
    a_parts = np.array_split(a, split_k, axis=1)
    b_parts = np.array_split(b, split_k, axis=0)
    partials = [ap.astype(np.float32) @ bp.astype(np.float32) for ap, bp in zip(a_parts, b_parts)]
    return sum(partials), partials


def splitk_extra_bytes(m: int, n: int, split_k: int, bytes_per_elem: int = 4) -> int:
    """Approximate extra traffic to write/read partial accumulators."""
    # Write split_k partials, read split_k partials, write final once.
    return (2 * split_k + 1) * m * n * bytes_per_elem


def smoke_test() -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(size=(8,101)).astype(np.float16)
    b = rng.normal(size=(101,9)).astype(np.float16)
    got, partials = splitk_gemm(a,b,4)
    assert len(partials) == 4
    assert np.allclose(got, a.astype(np.float32) @ b.astype(np.float32), atol=1e-2)
    assert splitk_extra_bytes(2,3,4) == 216
