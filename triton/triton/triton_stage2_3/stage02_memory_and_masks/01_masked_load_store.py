"""Stage 2.1 — Masked load and store.

Boundary checks are central to Triton.

The common pattern is:

    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, y, mask=mask)

Why `other=0.0`?
    Invalid lanes still need some value inside the vector expression. For elementwise kernels, zero is usually
    harmless. For max reductions, you often use `-inf`. For min reductions, `+inf`.
"""
from __future__ import annotations
import numpy as np


def masked_load(x: np.ndarray, offsets: np.ndarray, mask: np.ndarray, other: float = 0.0) -> np.ndarray:
    """NumPy reference for tl.load(ptr + offsets, mask=mask, other=other)."""
    out = np.full(offsets.shape, other, dtype=x.dtype)
    out[mask] = x[offsets[mask]]
    return out


def masked_store(out: np.ndarray, offsets: np.ndarray, values: np.ndarray, mask: np.ndarray) -> None:
    """NumPy reference for tl.store(ptr + offsets, values, mask=mask)."""
    out[offsets[mask]] = values[mask]


def copy_blocked(x: np.ndarray, block: int = 256) -> np.ndarray:
    """Copy a 1D tensor using Triton-style blocks and masks."""
    out = np.empty_like(x)
    n_programs = (x.size + block - 1) // block
    for pid in range(n_programs):
        offsets = pid * block + np.arange(block, dtype=np.int64)
        mask = offsets < x.size
        vals = masked_load(x, offsets, mask, other=0.0)
        masked_store(out, offsets, vals, mask)
    return out


def smoke_test() -> None:
    x = np.arange(10, dtype=np.float32)
    assert np.allclose(copy_blocked(x, block=4), x)
    offsets = np.array([8, 9, 10, 11])
    mask = offsets < x.size
    loaded = masked_load(x, offsets, mask, other=-1.0)
    assert loaded.tolist() == [8.0, 9.0, -1.0, -1.0]
