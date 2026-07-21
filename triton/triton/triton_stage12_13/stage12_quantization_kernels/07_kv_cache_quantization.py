from __future__ import annotations
"""Stage 12.7 — KV cache quantization.

KV cache can dominate inference memory. Quantizing K/V reduces memory, but can affect attention quality.

Common approaches:
    - per-token int8 scale for K and V
    - per-head scale
    - fp8 KV cache
    - mixed precision: quantized V or K only in some systems

This file implements per-token symmetric int8 quantization for [S,D].
"""

import numpy as np


def quantize_kv_per_token(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # x [S,D]
    max_abs = np.max(np.abs(x.astype(np.float32)), axis=1, keepdims=True)
    scale = np.maximum(max_abs / 127.0, 1e-8).astype(np.float32)
    q = np.clip(np.round(x.astype(np.float32) / scale), -127, 127).astype(np.int8)
    return q, scale.reshape(-1)


def dequantize_kv_per_token(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * scale.astype(np.float32)[:, None]


def kv_quantized_bytes(seq_len: int, head_dim: int, scale_bytes: int = 4) -> int:
    return seq_len * head_dim * 1 + seq_len * scale_bytes


def smoke_test() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(size=(10, 8)).astype(np.float32)
    q, s = quantize_kv_per_token(x)
    xh = dequantize_kv_per_token(q, s)
    assert q.dtype == np.int8 and s.shape == (10,)
    assert xh.shape == x.shape
    assert kv_quantized_bytes(10, 8) == 120
