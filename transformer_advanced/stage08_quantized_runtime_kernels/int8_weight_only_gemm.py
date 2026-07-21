"""
INT8 weight-only GEMM reference.

Runtime pattern:
    Activation X is FP16/FP32.
    Weight W is stored as INT8 plus scale per output channel or per group.
    During GEMM, weight values are dequantized or fused into the accumulation path.

This is common in LLM inference because weights dominate model memory.
"""
from __future__ import annotations
import numpy as np


def quantize_weight_per_channel(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize W[Din,Dout] per output channel."""
    max_abs = np.max(np.abs(w), axis=0)
    scale = np.where(max_abs == 0, 1.0, max_abs / 127.0).astype(np.float32)
    q = np.round(w / scale[None, :]).clip(-127, 127).astype(np.int8)
    return q, scale


def int8_weight_only_gemm(x: np.ndarray, q_w: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """
    Compute X @ dequant(W).

    Production kernels fuse dequantization with dot product and may use tensor cores with INT8 MMA.
    """
    w_deq = q_w.astype(np.float32) * scale[None, :]
    return x.astype(np.float32) @ w_deq


def max_quantization_error(w: np.ndarray) -> float:
    q, s = quantize_weight_per_channel(w)
    return float(np.max(np.abs(w - q.astype(np.float32) * s[None, :])))


def smoke_test() -> None:
    rng = np.random.default_rng(8)
    x = rng.normal(size=(4,16)).astype(np.float32)
    w = rng.normal(size=(16,7)).astype(np.float32)
    q, s = quantize_weight_per_channel(w)
    y = int8_weight_only_gemm(x,q,s)
    assert y.shape == (4,7)
    assert max_quantization_error(w) < 0.05
