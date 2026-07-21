from __future__ import annotations
"""Stage 5.0 — LayerNorm Triton kernel.

LayerNorm formula for each row x:

    mean = sum(x) / N
    var = sum((x - mean)^2) / N
    y = (x - mean) / sqrt(var + eps) * gamma + beta

Program mapping:
    One Triton program handles one row.

Why this matters:
    LayerNorm appears in Transformer blocks and is a common optimization/interview kernel.

Performance notes:
    - Reads x, gamma, beta and writes y.
    - Computes two reductions per row: mean and variance.
    - Often memory-bandwidth sensitive for moderate hidden sizes.
    - For fp16/bf16 inputs, compute mean/variance in fp32.

Limitations of this teaching kernel:
    - Hidden size N must be <= BLOCK.
    - A production kernel may use warp-level details, persistent programs, or fused residual paths.
"""

import time
import numpy as np

try:
    import torch
    import triton
    import triton.language as tl
except Exception:
    torch = triton = tl = None


def layernorm_reference(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    x32 = x.astype(np.float32)
    mean = np.mean(x32, axis=1, keepdims=True)
    var = np.mean((x32 - mean) ** 2, axis=1, keepdims=True)
    return (x32 - mean) / np.sqrt(var + eps) * gamma.astype(np.float32) + beta.astype(np.float32)


def layernorm_program_simulation(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5, block: int | None = None) -> np.ndarray:
    m, n = x.shape
    block = block or (1 << (n - 1).bit_length())
    out = np.empty((m, n), dtype=np.float32)
    for row in range(m):
        cols = np.arange(block)
        mask = cols < n
        safe = np.where(mask, cols, 0)
        vals = np.where(mask, x[row, safe].astype(np.float32), 0.0)
        mean = np.sum(vals) / n
        centered = np.where(mask, vals - mean, 0.0)
        var = np.sum(centered * centered) / n
        normed = centered / np.sqrt(var + eps)
        out[row, :] = normed[:n] * gamma.astype(np.float32) + beta.astype(np.float32)
    return out


if tl is not None:
    @triton.jit
    def layernorm_kernel(x_ptr, gamma_ptr, beta_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, stride_xm: tl.constexpr, stride_om: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / N
        centered = tl.where(mask, x - mean, 0.0)
        var = tl.sum(centered * centered, axis=0) / N
        y = centered * tl.rsqrt(var + eps) * gamma + beta
        tl.store(out_ptr + row * stride_om + cols, y, mask=mask)


def layernorm_triton(x, gamma, beta, eps: float = 1e-5, block: int | None = None):
    if triton is None or torch is None:
        raise RuntimeError('Triton/Torch not available')
    M, N = x.shape
    block = block or (1 << (N - 1).bit_length())
    out = torch.empty_like(x, dtype=torch.float32)
    layernorm_kernel[(M,)](x, gamma, beta, out, M, N, x.stride(0), out.stride(0), eps, BLOCK=block)
    return out


def benchmark_stub(m: int = 512, n: int = 1024) -> dict[str, float]:
    rng = np.random.default_rng(6)
    x = rng.normal(size=(m, n)).astype(np.float32)
    gamma = rng.normal(size=(n,)).astype(np.float32)
    beta = rng.normal(size=(n,)).astype(np.float32)
    t0 = time.perf_counter(); out = layernorm_program_simulation(x, gamma, beta); t1 = time.perf_counter()
    assert np.allclose(out, layernorm_reference(x, gamma, beta), rtol=1e-5, atol=1e-5)
    return {'cpu_seconds': t1 - t0, 'approx_bytes_read_write': float((3 * m * n + 2 * n) * 4)}


def smoke_test() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(9, 17)).astype(np.float32)
    gamma = rng.normal(size=(17,)).astype(np.float32)
    beta = rng.normal(size=(17,)).astype(np.float32)
    out = layernorm_program_simulation(x, gamma, beta, block=32)
    assert np.allclose(out, layernorm_reference(x, gamma, beta), rtol=1e-5, atol=1e-5)

if __name__ == '__main__':
    print(benchmark_stub())
