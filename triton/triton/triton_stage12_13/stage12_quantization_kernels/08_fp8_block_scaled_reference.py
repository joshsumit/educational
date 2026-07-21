from __future__ import annotations
"""Stage 12.8 — FP8 block-scaled reference.

This is a simplified educational FP8-like block-scaled format, not an exact hardware FP8 encoding.

Idea:
    split tensor into blocks
    for each block, choose scale based on max abs
    store low-precision integer code plus block scale

This prepares for FP8/block-scaled matmul reasoning without depending on hardware-specific FP8 types.
"""

import numpy as np


def block_scaled_quantize(x: np.ndarray, block: int = 32, qmax: int = 127) -> tuple[np.ndarray, np.ndarray]:
    flat = x.reshape(-1).astype(np.float32)
    num_blocks = (flat.size + block - 1) // block
    q = np.zeros_like(flat, dtype=np.int8)
    scales = np.empty((num_blocks,), dtype=np.float32)
    for b in range(num_blocks):
        s = b * block; e = min(s + block, flat.size)
        max_abs = float(np.max(np.abs(flat[s:e]))) if e > s else 0.0
        scale = max(max_abs / qmax, 1e-8)
        scales[b] = scale
        q[s:e] = np.clip(np.round(flat[s:e] / scale), -qmax, qmax).astype(np.int8)
    return q.reshape(x.shape), scales


def block_scaled_dequantize(q: np.ndarray, scales: np.ndarray, block: int = 32) -> np.ndarray:
    flat = q.reshape(-1).astype(np.float32)
    out = np.empty_like(flat, dtype=np.float32)
    for b, scale in enumerate(scales):
        s = b * block; e = min(s + block, flat.size)
        out[s:e] = flat[s:e] * scale
    return out.reshape(q.shape)


def smoke_test() -> None:
    rng = np.random.default_rng(6)
    x = rng.normal(size=(5, 9)).astype(np.float32)
    q, scales = block_scaled_quantize(x, block=8)
    xh = block_scaled_dequantize(q, scales, block=8)
    assert q.shape == x.shape and scales.ndim == 1
    assert np.max(np.abs(x - xh)) < 0.05
