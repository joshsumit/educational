"""Block vectors: the core Triton mental model.

In Triton, expressions like `offs = pid * BLOCK + tl.arange(0, BLOCK)` create a vector of offsets.
Then `mask = offs < n` guards boundary elements.
"""
from __future__ import annotations
import numpy as np

def arange_block(block_size: int) -> np.ndarray:
    return np.arange(block_size, dtype=np.int64)

def offsets_1d(pid: int, block_size: int) -> np.ndarray:
    return pid * block_size + arange_block(block_size)

def masks_1d(pid: int, n: int, block_size: int) -> np.ndarray:
    return offsets_1d(pid, block_size) < n

def offsets_2d(pid_m: int, pid_n: int, block_m: int, block_n: int) -> tuple[np.ndarray,np.ndarray]:
    rows = pid_m*block_m + np.arange(block_m)[:,None]
    cols = pid_n*block_n + np.arange(block_n)[None,:]
    return rows, cols

def smoke_test() -> None:
    assert np.array_equal(offsets_1d(2,4), np.array([8,9,10,11]))
    assert np.array_equal(masks_1d(2,10,4), np.array([True, True, False, False]))
    r,c=offsets_2d(1,2,2,3); assert r.shape==(2,1) and c.shape==(1,3)
