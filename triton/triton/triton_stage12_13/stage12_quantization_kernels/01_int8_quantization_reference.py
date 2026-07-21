from __future__ import annotations
"""Stage 12.1 — INT8 quantization reference.

This file implements per-tensor and per-channel symmetric int8 quantization.

Symmetric int8:
    scale = max(abs(x)) / 127
    q = clip(round(x / scale), -127, 127)
    x_hat = q * scale

Why preserve NumPy reference:
    - correctness oracle
    - measures quantization error
    - explains scale granularity clearly
"""

import numpy as np


def symmetric_int8_quantize_per_tensor(x: np.ndarray) -> tuple[np.ndarray, np.float32]:
    max_abs = float(np.max(np.abs(x.astype(np.float32))))
    scale = np.float32(max(max_abs / 127.0, 1e-8))
    q = np.clip(np.round(x.astype(np.float32) / scale), -127, 127).astype(np.int8)
    return q, scale


def symmetric_int8_dequantize(q: np.ndarray, scale: np.ndarray | float) -> np.ndarray:
    return q.astype(np.float32) * np.asarray(scale, dtype=np.float32)


def symmetric_int8_quantize_per_channel(x: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    max_abs = np.max(np.abs(x.astype(np.float32)), axis=axis, keepdims=True)
    scale = np.maximum(max_abs / 127.0, 1e-8).astype(np.float32)
    q = np.clip(np.round(x.astype(np.float32) / scale), -127, 127).astype(np.int8)
    return q, np.squeeze(scale, axis=axis)


def max_abs_error(x: np.ndarray, x_hat: np.ndarray) -> float:
    return float(np.max(np.abs(x.astype(np.float32) - x_hat.astype(np.float32))))


def smoke_test() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, 7)).astype(np.float32)
    q, s = symmetric_int8_quantize_per_tensor(x)
    xh = symmetric_int8_dequantize(q, s)
    assert q.dtype == np.int8
    assert max_abs_error(x, xh) < 0.05
    qc, sc = symmetric_int8_quantize_per_channel(x, axis=1)
    assert qc.shape == x.shape and sc.shape == (8,)
