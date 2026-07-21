"""
Double buffering for GEMM mainloops.

Two shared-memory buffers are used:
    buffer 0: compute current tile
    buffer 1: load next tile
Then they swap roles.

With more modern kernels, this generalizes to multi-stage pipelines.
"""
from __future__ import annotations
import numpy as np


def double_buffered_gemm(a: np.ndarray, b: np.ndarray, block_k: int = 32) -> np.ndarray:
    """
    Matrix multiply with explicit two-buffer K loop.

    This is not faster than NumPy. It exposes the dataflow of a GPU kernel mainloop.
    """
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError('inner dimensions must match')
    out = np.zeros((m, n), dtype=np.float32)
    buffers = [None, None]
    num_tiles = (k + block_k - 1) // block_k

    # Prologue: load tile 0.
    buffers[0] = (a[:, 0:block_k].astype(np.float32), b[0:block_k, :].astype(np.float32))

    for tile in range(num_tiles):
        compute_buffer = tile % 2
        load_buffer = 1 - compute_buffer
        next_tile = tile + 1
        if next_tile < num_tiles:
            k0 = next_tile * block_k
            buffers[load_buffer] = (a[:, k0:k0+block_k].astype(np.float32), b[k0:k0+block_k, :].astype(np.float32))
        a_tile, b_tile = buffers[compute_buffer]
        out += a_tile @ b_tile
    return out


def smoke_test() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(size=(9,25)).astype(np.float16)
    b = rng.normal(size=(25,7)).astype(np.float16)
    assert np.allclose(double_buffered_gemm(a,b,8), a.astype(np.float32) @ b.astype(np.float32), atol=1e-2)
