from __future__ import annotations
try:
    import torch
    import triton
    import triton.language as tl
except Exception:
    torch = triton = tl = None

"""Stage 3.0 — Vector add, the first real Triton kernel.

Formula:
    out[i] = x[i] + y[i]

What this teaches:
    - one program owns BLOCK elements
    - offsets are built from program_id and arange
    - masks protect the final partial block
    - loads and stores are vectorized at the program level
"""
import time
import numpy as np


def vector_add_reference(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x + y


def vector_add_blocked_reference(x: np.ndarray, y: np.ndarray, block: int = 1024) -> np.ndarray:
    out = np.empty_like(x)
    for pid in range((x.size + block - 1) // block):
        offsets = pid * block + np.arange(block)
        mask = offsets < x.size
        out[offsets[mask]] = x[offsets[mask]] + y[offsets[mask]]
    return out


if tl is not None:
    @triton.jit
    def vector_add_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, x + y, mask=mask)


def vector_add_triton(x, y, block: int = 1024):
    if triton is None or torch is None:
        raise RuntimeError('Triton/Torch not available')
    out = torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), block),)
    vector_add_kernel[grid](x, y, out, x.numel(), BLOCK=block)
    return out


def benchmark_stub(n: int = 1_000_000, block: int = 1024) -> dict[str, float]:
    x = np.random.default_rng(0).normal(size=n).astype(np.float32)
    y = np.random.default_rng(1).normal(size=n).astype(np.float32)
    t0 = time.perf_counter(); out = vector_add_blocked_reference(x, y, block); t1 = time.perf_counter()
    assert np.allclose(out, x + y)
    bytes_moved = 3 * n * 4
    return {'cpu_seconds': t1 - t0, 'approx_bytes_moved': float(bytes_moved)}


def smoke_test() -> None:
    x = np.arange(33, dtype=np.float32); y = 2 * x
    assert np.allclose(vector_add_blocked_reference(x, y, block=8), vector_add_reference(x, y))


if __name__ == '__main__':
    print(benchmark_stub())
