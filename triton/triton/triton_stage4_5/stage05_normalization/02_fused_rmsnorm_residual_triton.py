from __future__ import annotations
"""Stage 5.2 — Fused residual add + RMSNorm Triton kernel.

Common LLM pattern:

    z = x + residual
    y = RMSNorm(z, weight)

Unfused path:
    1. read x
    2. read residual
    3. write z temporary
    4. read z again
    5. read weight
    6. write y

Fused path:
    1. read x
    2. read residual
    3. compute z in registers
    4. reduce z^2
    5. read weight
    6. write y

The fused path avoids the temporary global-memory round trip.

This is a very interview-relevant inference kernel because residual+norm appears at every Transformer layer.
"""

import time
import numpy as np

try:
    import torch
    import triton
    import triton.language as tl
except Exception:
    torch = triton = tl = None


def fused_rmsnorm_residual_reference(x: np.ndarray, residual: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    z = x.astype(np.float32) + residual.astype(np.float32)
    rms = np.sqrt(np.mean(z * z, axis=1, keepdims=True) + eps)
    return z / rms * weight.astype(np.float32)


def fused_rmsnorm_residual_program_simulation(x: np.ndarray, residual: np.ndarray, weight: np.ndarray, eps: float = 1e-6, block: int | None = None) -> np.ndarray:
    m, n = x.shape
    block = block or (1 << (n - 1).bit_length())
    out = np.empty((m, n), dtype=np.float32)
    for row in range(m):
        cols = np.arange(block)
        mask = cols < n
        safe = np.where(mask, cols, 0)
        xv = np.where(mask, x[row, safe].astype(np.float32), 0.0)
        rv = np.where(mask, residual[row, safe].astype(np.float32), 0.0)
        z = xv + rv
        mean_square = np.sum(z * z) / n
        inv_rms = 1.0 / np.sqrt(mean_square + eps)
        out[row, :] = z[:n] * inv_rms * weight.astype(np.float32)
    return out


def memory_traffic_bytes(m: int, n: int, bytes_per_element: int = 4) -> dict[str, int]:
    """Simple traffic model for fused vs unfused residual+RMSNorm.

    Unfused approximate element traffic:
        read x, read residual, write z, read z, read weight, write y = 6 logical element streams

    Fused approximate element traffic:
        read x, read residual, read weight, write y = 4 logical element streams
    """
    elems = m * n
    unfused = 6 * elems * bytes_per_element
    fused = 4 * elems * bytes_per_element
    return {'unfused_bytes': unfused, 'fused_bytes': fused, 'bytes_saved': unfused - fused}


if tl is not None:
    @triton.jit
    def fused_rmsnorm_residual_kernel(x_ptr, residual_ptr, weight_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, stride_xm: tl.constexpr, stride_rm: tl.constexpr, stride_om: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(residual_ptr + row * stride_rm + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        z = x + r
        mean_square = tl.sum(z * z, axis=0) / N
        inv_rms = tl.rsqrt(mean_square + eps)
        y = z * inv_rms * w
        tl.store(out_ptr + row * stride_om + cols, y, mask=mask)


def fused_rmsnorm_residual_triton(x, residual, weight, eps: float = 1e-6, block: int | None = None):
    if triton is None or torch is None:
        raise RuntimeError('Triton/Torch not available')
    M, N = x.shape
    block = block or (1 << (N - 1).bit_length())
    out = torch.empty_like(x, dtype=torch.float32)
    fused_rmsnorm_residual_kernel[(M,)](x, residual, weight, out, M, N, x.stride(0), residual.stride(0), out.stride(0), eps, BLOCK=block)
    return out


def benchmark_stub(m: int = 512, n: int = 1024) -> dict[str, float | int]:
    rng = np.random.default_rng(10)
    x = rng.normal(size=(m, n)).astype(np.float32)
    residual = rng.normal(size=(m, n)).astype(np.float32)
    weight = rng.normal(size=(n,)).astype(np.float32)
    t0 = time.perf_counter(); out = fused_rmsnorm_residual_program_simulation(x, residual, weight); t1 = time.perf_counter()
    assert np.allclose(out, fused_rmsnorm_residual_reference(x, residual, weight), rtol=1e-5, atol=1e-5)
    return {'cpu_seconds': t1 - t0, **memory_traffic_bytes(m, n)}


def smoke_test() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(5, 21)).astype(np.float32)
    residual = rng.normal(size=(5, 21)).astype(np.float32)
    weight = rng.normal(size=(21,)).astype(np.float32)
    out = fused_rmsnorm_residual_program_simulation(x, residual, weight, block=32)
    exp = fused_rmsnorm_residual_reference(x, residual, weight)
    assert np.allclose(out, exp, rtol=1e-5, atol=1e-5)
    assert memory_traffic_bytes(1, 10)['bytes_saved'] == 80

if __name__ == '__main__':
    print(benchmark_stub())
