"""Stage 0.5 — Common Triton primitives, explained through CPU analogues.

This file gives CPU equivalents for the Triton primitives you will repeatedly see in real kernels.

Primitive mapping:

    tl.program_id(axis)
        -> current program coordinate along a grid axis

    tl.arange(0, BLOCK)
        -> vector [0, 1, 2, ..., BLOCK-1]

    tl.load(ptr + offsets, mask=mask, other=value)
        -> masked gather from memory

    tl.store(ptr + offsets, values, mask=mask)
        -> masked scatter to memory

    tl.where(condition, a, b)
        -> elementwise select

    tl.max(x, axis=0), tl.sum(x, axis=0)
        -> block-vector reductions

    tl.dot(a, b)
        -> matrix multiply primitive used inside tiled matmul kernels
"""
from __future__ import annotations

import numpy as np


def tl_arange(start: int, stop: int) -> np.ndarray:
    """CPU analogue of tl.arange(start, stop)."""
    return np.arange(start, stop, dtype=np.int64)


def tl_where(condition: np.ndarray, a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    """CPU analogue of tl.where."""
    return np.where(condition, a, b)


def masked_load_1d(x: np.ndarray, offsets: np.ndarray, mask: np.ndarray, other: float = 0.0) -> np.ndarray:
    """CPU analogue of tl.load for a 1D pointer plus offsets."""
    result = np.full(offsets.shape, other, dtype=x.dtype)
    result[mask] = x[offsets[mask]]
    return result


def masked_store_1d(out: np.ndarray, offsets: np.ndarray, values: np.ndarray, mask: np.ndarray) -> None:
    """CPU analogue of tl.store for a 1D pointer plus offsets."""
    out[offsets[mask]] = values[mask]


def block_sum(values: np.ndarray) -> float:
    """CPU analogue of tl.sum over a block vector."""
    return float(np.sum(values))


def block_max(values: np.ndarray) -> float:
    """CPU analogue of tl.max over a block vector."""
    return float(np.max(values))


def smoke_test() -> None:
    x = np.arange(6, dtype=np.float32)
    offsets = np.array([4, 5, 6, 7])
    mask = offsets < x.size
    loaded = masked_load_1d(x, offsets, mask, other=-1.0)
    assert loaded.tolist() == [4.0, 5.0, -1.0, -1.0]
    out = np.zeros_like(x)
    masked_store_1d(out, offsets, loaded, mask)
    assert out.tolist() == [0.0, 0.0, 0.0, 0.0, 4.0, 5.0]
    assert block_sum(np.array([1, 2, 3])) == 6.0
    assert block_max(np.array([1, 5, 3])) == 5.0
