from __future__ import annotations
"""Stage 5.1 — RMSNorm Triton kernel.

RMSNorm formula for each row x:

    rms = sqrt(mean(x^2) + eps)
    y = x / rms * weight

Difference from LayerNorm:
    - RMSNorm does not subtract the mean.
    - It usually has only one learned scale vector, often called weight.
    - It is cheaper than LayerNorm: one reduction instead of mean+variance.

Why this matters:
    RMSNorm is common in modern LLMs. It is a high-ROI kernel for inference interviews.
"""

import time
import numpy as np

try:
    import torch
    import triton
    import triton.language as tl
except Exception:
    torch = triton = tl = None


def rmsnorm_reference(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x32 = x.astype(np.float32)
    rms = np.sqrt(np.mean(x32 * x32, axis=1, keepdims=True) + eps)
    return x32 / rms * weight.astype(np.float32)


def rmsnorm_program_simulation(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6, block: int | None = None) -> np.ndarray:
    m, n = x.shape
    block = block or (1 << (n - 1).bit_length())
    out = np.empty((m, n), dtype=np.float32)
    for row in range(m):
        cols = np.arange(block)
        mask = cols < n
        safe = np.where(mask, cols, 0)
        vals = np.where(mask, x[row, safe].astype(np.float32), 0.0)
        mean_square = np.sum(vals * vals) / n
        inv_rms = 1.0 / np.sqrt(mean_square + eps)
        out[row, :] = vals[:n] * inv_rms * weight.astype(np.float32)
    return out


if tl is not None:
    @triton.jit
    def rmsnorm_kernel(x_ptr, weight_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, stride_xm: tl.constexpr, stride_om: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        mean_square = tl.sum(x * x, axis=0) / N
        inv_rms = tl.rsqrt(mean_square + eps)
        y = x * inv_rms * w
        tl.store(out_ptr + row * stride_om + cols, y, mask=mask)


def rmsnorm_triton(x, weight, eps: float = 1e-6, block: int | None = None):
    if triton is None or torch is None:
        raise RuntimeError('Triton/Torch not available')
    M, N = x.shape
    block = block or (1 << (N - 1).bit_length())
    out = torch.empty_like(x, dtype=torch.float32)
    rmsnorm_kernel[(M,)](x, weight, out, M, N, x.stride(0), out.stride(0), eps, BLOCK=block)
    return out


def benchmark_stub(m: int = 512, n: int = 1024) -> dict[str, float]:
    rng = np.random.default_rng(8)
    x = rng.normal(size=(m, n)).astype(np.float32)
    weight = rng.normal(size=(n,)).astype(np.float32)
    t0 = time.perf_counter(); out = rmsnorm_program_simulation(x, weight); t1 = time.perf_counter()
    assert np.allclose(out, rmsnorm_reference(x, weight), rtol=1e-5, atol=1e-5)
    return {'cpu_seconds': t1 - t0, 'approx_bytes_read_write': float((2 * m * n + n) * 4)}


def smoke_test() -> None:
    rng = np.random.default_rng(9)
    x = rng.normal(size=(7, 19)).astype(np.float32)
    weight = rng.normal(size=(19,)).astype(np.float32)
    out = rmsnorm_program_simulation(x, weight, block=32)
    assert np.allclose(out, rmsnorm_reference(x, weight), rtol=1e-5, atol=1e-5)

if __name__ == '__main__':
    print(benchmark_stub())
