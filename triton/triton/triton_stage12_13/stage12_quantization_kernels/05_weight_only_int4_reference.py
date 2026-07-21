from __future__ import annotations
"""Stage 12.5 — Weight-only INT4 matmul reference.

Weight-only quantization keeps activations in fp16/fp32 and stores weights in int4.

Formula:
    W_q int4 [K,N]
    scale per output channel N or per group
    W_hat[:, n] = W_q[:, n] * scale[n]
    C = A_fp @ W_hat

This saves weight bandwidth, but the kernel must unpack/dequantize weights efficiently.
"""

import numpy as np


def quantize_weight_int4_per_channel(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_abs = np.max(np.abs(w.astype(np.float32)), axis=0, keepdims=True)
    scale = np.maximum(max_abs / 7.0, 1e-8).astype(np.float32)
    q = np.clip(np.round(w.astype(np.float32) / scale), -8, 7).astype(np.int8)
    return q, scale.reshape(-1)


def dequantize_weight_int4_per_channel(wq: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return wq.astype(np.float32) * scale.astype(np.float32)[None, :]


def weight_only_int4_matmul_reference(a: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    wq, scale = quantize_weight_int4_per_channel(w)
    what = dequantize_weight_int4_per_channel(wq, scale)
    out = a.astype(np.float32) @ what
    return out, {'weight_q': wq, 'scale': scale}


def smoke_test() -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(size=(8, 16)).astype(np.float32)
    w = rng.normal(size=(16, 5)).astype(np.float32)
    out, meta = weight_only_int4_matmul_reference(a, w)
    assert out.shape == (8, 5)
    assert meta['weight_q'].dtype == np.int8
