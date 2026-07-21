"""Pointer arithmetic and strides for Triton kernels.

A tensor element address is base + row * stride_row + col * stride_col.
Triton kernels often receive raw pointers plus strides.
"""
from __future__ import annotations
import numpy as np

def row_major_offsets(rows: np.ndarray, cols: np.ndarray, stride_row: int, stride_col: int = 1) -> np.ndarray:
    return rows * stride_row + cols * stride_col

def make_tile_offsets(row_start: int, col_start: int, block_m: int, block_n: int, stride_row: int) -> np.ndarray:
    rows = row_start + np.arange(block_m)[:,None]
    cols = col_start + np.arange(block_n)[None,:]
    return row_major_offsets(rows, cols, stride_row)

def coalescing_score(offsets: np.ndarray) -> float:
    """Simple score: fraction of adjacent flat offsets that are contiguous."""
    flat=offsets.reshape(-1)
    if flat.size<2: return 1.0
    return float(np.mean(np.diff(flat)==1))

def smoke_test() -> None:
    offs=make_tile_offsets(0,0,2,4,8)
    assert offs.tolist()==[[0,1,2,3],[8,9,10,11]]
    assert coalescing_score(np.arange(8))==1.0
