"""Stage 1.1 — Block vectors.

The most important Triton expression for beginners is:

    offs = pid * BLOCK + tl.arange(0, BLOCK)

This creates a vector of offsets, not a scalar index.

Example:

    pid = 2
    BLOCK = 8

    tl.arange(0, BLOCK) -> [0, 1, 2, 3, 4, 5, 6, 7]
    pid * BLOCK       -> 16
    offsets           -> [16, 17, 18, 19, 20, 21, 22, 23]

Why this matters:
    Triton programs express operations over vectors/tiles. The compiler maps those vector operations onto
    lower-level GPU execution.
"""
from __future__ import annotations

import numpy as np


def arange_block(block_size: int) -> np.ndarray:
    """Return [0, 1, ..., block_size-1]."""
    if block_size <= 0:
        raise ValueError('block_size must be positive')
    return np.arange(block_size, dtype=np.int64)


def offsets_1d(pid: int, block_size: int) -> np.ndarray:
    """Return the logical offsets for one 1D Triton program."""
    return pid * block_size + arange_block(block_size)


def offsets_2d(pid_m: int, pid_n: int, block_m: int, block_n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return row and column offset vectors for one 2D tile.

    rows has shape [BLOCK_M, 1]
    cols has shape [1, BLOCK_N]

    Later, pointer arithmetic will combine them as:
        base + rows * stride_row + cols * stride_col
    """
    rows = pid_m * block_m + np.arange(block_m, dtype=np.int64)[:, None]
    cols = pid_n * block_n + np.arange(block_n, dtype=np.int64)[None, :]
    return rows, cols


def tile_coordinates(pid_m: int, pid_n: int, block_m: int, block_n: int) -> np.ndarray:
    """Return a [BLOCK_M, BLOCK_N, 2] array of (row, col) coordinates."""
    rows, cols = offsets_2d(pid_m, pid_n, block_m, block_n)
    rr = np.broadcast_to(rows, (block_m, block_n))
    cc = np.broadcast_to(cols, (block_m, block_n))
    return np.stack([rr, cc], axis=-1)


def smoke_test() -> None:
    assert np.array_equal(arange_block(4), np.array([0, 1, 2, 3]))
    assert np.array_equal(offsets_1d(2, 4), np.array([8, 9, 10, 11]))
    rows, cols = offsets_2d(1, 2, 2, 3)
    assert rows.tolist() == [[2], [3]]
    assert cols.tolist() == [[6, 7, 8]]
    coords = tile_coordinates(1, 2, 2, 3)
    assert coords.shape == (2, 3, 2)
    assert coords[0, 0].tolist() == [2, 6]
