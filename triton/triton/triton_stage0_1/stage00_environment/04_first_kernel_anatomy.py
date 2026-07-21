"""Stage 0.4 — Anatomy of the first Triton kernel.

This file does not execute Triton. It captures the structure of a real vector-add kernel and then simulates
the same ownership logic in Python.

Canonical Triton vector-add kernel:

    @triton.jit
    def vector_add_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        y = tl.load(y_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, x + y, mask=mask)

Line-by-line:

    @triton.jit
        Compile this Python function into a GPU kernel.

    pid = tl.program_id(0)
        Get the id of this program along launch dimension 0.

    offs = pid * BLOCK + tl.arange(0, BLOCK)
        Build the vector of logical element offsets owned by this program.

    mask = offs < n
        Mark which offsets are valid. This protects the final partial block.

    tl.load(..., mask=mask, other=0.0)
        Load only valid elements. Invalid lanes get `other`.

    tl.store(..., mask=mask)
        Store only valid output elements.
"""
from __future__ import annotations

import numpy as np


def vector_add_program_simulation(x: np.ndarray, y: np.ndarray, pid: int, block: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate one Triton program of vector add.

    Returns:
        offsets: logical offsets for the program
        mask: boolean validity mask
        values: computed values for all lanes, invalid lanes filled with 0
    """
    if x.shape != y.shape:
        raise ValueError('x and y must have the same shape')
    n = x.size
    offsets = pid * block + np.arange(block)
    mask = offsets < n

    # `safe_offsets` prevents NumPy from indexing out of bounds during the simulation.
    # In real Triton, tl.load uses the mask to avoid invalid memory access.
    safe_offsets = np.where(mask, offsets, 0)
    values = np.where(mask, x[safe_offsets] + y[safe_offsets], 0.0)
    return offsets, mask, values


def vector_add_full_simulation(x: np.ndarray, y: np.ndarray, block: int) -> np.ndarray:
    """Run a full CPU simulation of the Triton vector-add launch."""
    out = np.empty_like(x)
    n_programs = (x.size + block - 1) // block
    for pid in range(n_programs):
        offsets, mask, values = vector_add_program_simulation(x, y, pid, block)
        out[offsets[mask]] = values[mask]
    return out


def smoke_test() -> None:
    x = np.arange(10, dtype=np.float32)
    y = 10 * x
    offsets, mask, values = vector_add_program_simulation(x, y, pid=1, block=8)
    assert offsets.tolist() == [8, 9, 10, 11, 12, 13, 14, 15]
    assert mask.tolist() == [True, True, False, False, False, False, False, False]
    assert values[:2].tolist() == [88.0, 99.0]
    assert np.allclose(vector_add_full_simulation(x, y, block=8), x + y)
