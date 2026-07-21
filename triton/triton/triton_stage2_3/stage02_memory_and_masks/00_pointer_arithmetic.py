"""Stage 2.0 — Pointer arithmetic.

Triton kernels usually receive raw pointers plus strides. A pointer expression is not magic; it is an address
calculation.

For a row-major matrix X[M, N]:

    address(row, col) = base + row * stride_row + col * stride_col

For a contiguous row-major matrix:

    stride_row = N
    stride_col = 1

In real Triton you will see:

    offs = rows[:, None] * stride_row + cols[None, :] * stride_col
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

This file keeps the same logic in NumPy so the memory math can be tested CPU-only.
"""
from __future__ import annotations
import numpy as np


def row_major_offset(row: int, col: int, stride_row: int, stride_col: int = 1) -> int:
    """Return the flat offset of one element using explicit strides."""
    return row * stride_row + col * stride_col


def tile_offsets(row_start: int, col_start: int, block_m: int, block_n: int, stride_row: int, stride_col: int = 1) -> np.ndarray:
    """Return a [BLOCK_M, BLOCK_N] matrix of flat offsets for a tile.

    This is the exact shape style used in Triton matmul, softmax, layernorm, and attention kernels.
    """
    rows = row_start + np.arange(block_m, dtype=np.int64)[:, None]
    cols = col_start + np.arange(block_n, dtype=np.int64)[None, :]
    return rows * stride_row + cols * stride_col


def gather_tile(x_flat: np.ndarray, offsets: np.ndarray, mask: np.ndarray, other: float = 0.0) -> np.ndarray:
    """Masked gather using flat offsets.

    Real Triton equivalent:

        vals = tl.load(x_ptr + offsets, mask=mask, other=other)
    """
    out = np.full(offsets.shape, other, dtype=x_flat.dtype)
    out[mask] = x_flat[offsets[mask]]
    return out


def smoke_test() -> None:
    x = np.arange(4 * 6, dtype=np.float32).reshape(4, 6)
    offs = tile_offsets(1, 2, 2, 3, stride_row=6)
    assert offs.tolist() == [[8, 9, 10], [14, 15, 16]]
    vals = gather_tile(x.reshape(-1), offs, np.ones_like(offs, dtype=bool))
    assert vals.tolist() == [[8.0, 9.0, 10.0], [14.0, 15.0, 16.0]]
