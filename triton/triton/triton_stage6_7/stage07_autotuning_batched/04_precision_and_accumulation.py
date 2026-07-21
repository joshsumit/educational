from __future__ import annotations
"""Stage 7.4 — Precision and accumulation.

For AI matmul kernels, inputs may be fp32, fp16, bf16, fp8, int8, or int4. A frequent rule is:

    low precision inputs, higher precision accumulation

Examples:
    fp16 x fp16 -> fp32 accumulation, sometimes fp16 output
    bf16 x bf16 -> fp32 accumulation, sometimes bf16 output
    int8 x int8 -> int32 accumulation, then scale/dequantize

This stage is not implementing quantized matmul yet. It captures the reasoning needed before quantization stages.
"""

import numpy as np


def matmul_accumulate_fp32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a.astype(np.float32) @ b.astype(np.float32)


def matmul_accumulate_fp16_like(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Educational bad baseline: simulate lower precision accumulation by casting inputs.

    NumPy still may not exactly model hardware fp16 accumulation. Use this only as a conceptual comparison.
    """
    return a.astype(np.float16) @ b.astype(np.float16)


def max_abs_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32))))


def smoke_test() -> None:
    rng = np.random.default_rng(9)
    a = rng.normal(size=(16, 32)).astype(np.float32)
    b = rng.normal(size=(32, 8)).astype(np.float32)
    fp32 = matmul_accumulate_fp32(a, b)
    fp16_like = matmul_accumulate_fp16_like(a, b)
    assert fp32.shape == fp16_like.shape
    assert max_abs_error(fp16_like, fp32) >= 0.0
