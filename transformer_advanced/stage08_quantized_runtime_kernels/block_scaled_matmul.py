"""
Block-scaled matmul reference.

Block scaling is used by FP8 / low-bit formats and some custom accelerators.
A tensor is split into blocks; each block has a scale. Values are stored in a small format.

This file uses int8 as the storage format but the same structure applies to many block-scaled schemes.
"""
from __future__ import annotations
import numpy as np


def quantize_blocks(x: np.ndarray, block_cols: int = 32) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = x.shape
    num_blocks = (cols + block_cols - 1) // block_cols
    q = np.zeros_like(x, dtype=np.int8)
    scales = np.zeros((rows, num_blocks), dtype=np.float32)
    for r in range(rows):
        for b in range(num_blocks):
            c0 = b * block_cols
            block = x[r, c0:c0+block_cols]
            max_abs = np.max(np.abs(block)) if block.size else 0.0
            scale = 1.0 if max_abs == 0 else max_abs / 127.0
            scales[r, b] = scale
            q[r, c0:c0+block_cols] = np.round(block / scale).clip(-127,127).astype(np.int8)
    return q, scales


def dequantize_blocks(q: np.ndarray, scales: np.ndarray, block_cols: int = 32) -> np.ndarray:
    out = np.zeros_like(q, dtype=np.float32)
    for r in range(q.shape[0]):
        for b in range(scales.shape[1]):
            c0 = b * block_cols
            out[r, c0:c0+block_cols] = q[r, c0:c0+block_cols].astype(np.float32) * scales[r, b]
    return out


def block_scaled_matmul(a: np.ndarray, b: np.ndarray, block_cols: int = 32) -> np.ndarray:
    q_a, s_a = quantize_blocks(a.astype(np.float32), block_cols)
    q_b_t, s_b_t = quantize_blocks(b.T.astype(np.float32), block_cols)
    a_deq = dequantize_blocks(q_a, s_a, block_cols)
    b_deq = dequantize_blocks(q_b_t, s_b_t, block_cols).T
    return a_deq @ b_deq


def smoke_test() -> None:
    rng = np.random.default_rng(9)
    a = rng.normal(size=(5,40)).astype(np.float32)
    b = rng.normal(size=(40,6)).astype(np.float32)
    y = block_scaled_matmul(a,b,16)
    assert y.shape == (5,6)
    assert np.mean(np.abs(y - a @ b)) < 0.2
