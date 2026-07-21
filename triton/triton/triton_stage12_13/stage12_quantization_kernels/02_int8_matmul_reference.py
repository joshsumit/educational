from __future__ import annotations
"""Stage 12.2 — INT8 matmul reference.

Two common paths:

1. Quantize A and B to int8, do int32 accumulation, then dequantize:

    C_fp32 ~= (A_int8 @ B_int8) * scale_a * scale_b

2. Per-channel B scales:

    C[:, n] ~= sum_k A_q[:,k] * B_q[k,n] * scale_a * scale_b[n]

This file uses NumPy int32 accumulation to make the kernel math explicit.
"""

import numpy as np
import importlib
_q = importlib.import_module('stage12_quantization_kernels.01_int8_quantization_reference')
symmetric_int8_quantize_per_tensor = _q.symmetric_int8_quantize_per_tensor


def int8_matmul_per_tensor_reference(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    aq, ascale = symmetric_int8_quantize_per_tensor(a)
    bq, bscale = symmetric_int8_quantize_per_tensor(b)
    acc = aq.astype(np.int32) @ bq.astype(np.int32)
    out = acc.astype(np.float32) * ascale * bscale
    return out, {'a_scale': ascale, 'b_scale': bscale, 'acc_dtype': 'int32'}


def int8_matmul_given_quantized(aq: np.ndarray, bq: np.ndarray, a_scale: float, b_scale: float | np.ndarray) -> np.ndarray:
    acc = aq.astype(np.int32) @ bq.astype(np.int32)
    return acc.astype(np.float32) * np.asarray(a_scale, dtype=np.float32) * np.asarray(b_scale, dtype=np.float32)


def smoke_test() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(size=(16, 32)).astype(np.float32)
    b = rng.normal(size=(32, 8)).astype(np.float32)
    out, meta = int8_matmul_per_tensor_reference(a, b)
    assert out.shape == (16, 8)
    assert meta['acc_dtype'] == 'int32'
