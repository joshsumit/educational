"""Stage 1.2 — Masks.

Boundary masks are non-negotiable in Triton.

Example:

    N = 10
    BLOCK = 8
    pid = 1

    offsets = [8, 9, 10, 11, 12, 13, 14, 15]
    mask    = [T, T, F,  F,  F,  F,  F,  F]

Real Triton:

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, y, mask=mask)

Interview answer:
    Masks prevent out-of-bounds memory access for partial tiles, especially the last block in a 1D launch
    or edge tiles in a 2D matmul.
"""
from __future__ import annotations

import numpy as np


def mask_1d(offsets: np.ndarray, n: int) -> np.ndarray:
    """Return `offsets < n`."""
    return offsets < n


def masked_offsets(pid: int, n: int, block: int) -> tuple[np.ndarray, np.ndarray]:
    """Return offsets and validity mask for one program."""
    offsets = pid * block + np.arange(block, dtype=np.int64)
    return offsets, mask_1d(offsets, n)


def mask_2d(rows: np.ndarray, cols: np.ndarray, m: int, n: int) -> np.ndarray:
    """Return a 2D tile mask for matrix-like tensors.

    rows shape: [BLOCK_M, 1]
    cols shape: [1, BLOCK_N]
    output: [BLOCK_M, BLOCK_N]
    """
    return (rows < m) & (cols < n)


def invalid_count(mask: np.ndarray) -> int:
    """Count invalid/off-boundary lanes."""
    return int(mask.size - np.count_nonzero(mask))


def smoke_test() -> None:
    offsets, mask = masked_offsets(pid=1, n=10, block=8)
    assert offsets.tolist() == [8, 9, 10, 11, 12, 13, 14, 15]
    assert mask.tolist() == [True, True, False, False, False, False, False, False]
    rows = np.array([[2], [3], [4]])
    cols = np.array([[4, 5, 6]])
    m = mask_2d(rows, cols, m=4, n=6)
    assert m.tolist() == [[True, True, False], [True, True, False], [False, False, False]]
    assert invalid_count(m) == 5
