from __future__ import annotations
"""Stage 12.6 — Weight-only INT4 Triton skeleton.

Production W4A16 kernels are complex because they require:

    - packed int4 weight loads
    - nibble extraction
    - signed int4 conversion
    - per-channel or per-group scaling
    - efficient dot product with fp16/bf16 activations

This file gives a readable skeleton and CPU reference. The Triton skeleton shows the unpacking path, but it is
intentionally not advertised as a production-optimized kernel.
"""

import numpy as np

try:
    import torch
    import triton
    import triton.language as tl
except Exception:
    torch = triton = tl = None

import importlib
_i4 = importlib.import_module('stage12_quantization_kernels.04_int4_packing_reference')
pack_int4 = _i4.pack_int4
unpack_int4 = _i4.unpack_int4


def unpacked_weight_only_reference(a: np.ndarray, packed_w: np.ndarray, scale: np.ndarray, k: int, n: int) -> np.ndarray:
    wq = unpack_int4(packed_w, original_size=k * n).reshape(k, n)
    w = wq.astype(np.float32) * scale.astype(np.float32)[None, :]
    return a.astype(np.float32) @ w


if tl is not None:
    @triton.jit
    def extract_signed_int4(byte_vals, high: tl.constexpr):
        nib = (byte_vals >> 4) & 0x0F if high else byte_vals & 0x0F
        return tl.where(nib >= 8, nib - 16, nib)

    # A complete optimized W4A16 matmul is deferred to a later vendor-specific optimization stage.
    # This skeleton exists to show where packed loads, nibble extraction, and scales enter the kernel.


def smoke_test() -> None:
    rng = np.random.default_rng(4)
    a = rng.normal(size=(4, 6)).astype(np.float32)
    wq = rng.integers(-8, 8, size=(6, 5), dtype=np.int8)
    scale = np.ones((5,), dtype=np.float32) * 0.1
    packed = pack_int4(wq)
    out = unpacked_weight_only_reference(a, packed, scale, 6, 5)
    assert out.shape == (4, 5)

if __name__ == '__main__': smoke_test(); print('ok')
