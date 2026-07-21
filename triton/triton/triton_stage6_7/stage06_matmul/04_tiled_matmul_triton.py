from __future__ import annotations
"""Stage 6.4 — Tiled Triton matmul with tl.dot.

This is the first real matmul kernel pattern.

Program mapping:
    One Triton program computes one C tile of shape [BLOCK_M, BLOCK_N].

Inside one program:
    - Build row offsets for M.
    - Build column offsets for N.
    - Build K offsets for BLOCK_K.
    - Loop over K in BLOCK_K chunks.
    - Load A tile [BLOCK_M, BLOCK_K].
    - Load B tile [BLOCK_K, BLOCK_N].
    - Accumulate acc += tl.dot(A_tile, B_tile).
    - Store C tile with boundary mask.

Important masks:
    A mask: rows < M and k_offsets < K
    B mask: k_offsets < K and cols < N
    C mask: rows < M and cols < N

Interview notes:
    - This is where arithmetic intensity improves over elementwise kernels.
    - Larger tiles improve reuse but increase registers/shared resources.
    - BLOCK_K controls how much K is processed per inner-loop iteration.
    - tl.dot maps to efficient matrix multiply paths depending on dtype and backend.
"""

import time
import numpy as np

try:
    import torch
    import triton
    import triton.language as tl
except Exception:
    torch = triton = tl = None


def matmul_tiled_cpu(a: np.ndarray, b: np.ndarray, block_m: int = 16, block_n: int = 16, block_k: int = 32) -> np.ndarray:
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError('shape mismatch')
    c = np.zeros((m, n), dtype=np.float32)
    for m0 in range(0, m, block_m):
        for n0 in range(0, n, block_n):
            acc = np.zeros((min(block_m, m - m0), min(block_n, n - n0)), dtype=np.float32)
            for k0 in range(0, k, block_k):
                acc += a[m0:m0+block_m, k0:k0+block_k].astype(np.float32) @ b[k0:k0+block_k, n0:n0+block_n].astype(np.float32)
            c[m0:m0+acc.shape[0], n0:n0+acc.shape[1]] = acc
    return c


if tl is not None:
    @triton.jit
    def matmul_tiled_kernel(
        a_ptr, b_ptr, c_ptr,
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        stride_am: tl.constexpr, stride_ak: tl.constexpr,
        stride_bk: tl.constexpr, stride_bn: tl.constexpr,
        stride_cm: tl.constexpr, stride_cn: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k0 in range(0, K, BLOCK_K):
            a = tl.load(
                a_ptr + offs_m[:, None] * stride_am + (k0 + offs_k[None, :]) * stride_ak,
                mask=(offs_m[:, None] < M) & ((k0 + offs_k[None, :]) < K),
                other=0.0,
            )
            b = tl.load(
                b_ptr + (k0 + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn,
                mask=((k0 + offs_k[:, None]) < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            acc += tl.dot(a, b)
        tl.store(
            c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
            acc,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


def matmul_tiled_triton(a, b, block_m: int = 16, block_n: int = 16, block_k: int = 32):
    if triton is None or torch is None:
        raise RuntimeError('Triton/Torch not available')
    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise ValueError('shape mismatch')
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    matmul_tiled_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k)
    return c


def benchmark_stub(m: int = 256, n: int = 256, k: int = 256) -> dict[str, float]:
    rng = np.random.default_rng(3)
    a = rng.normal(size=(m, k)).astype(np.float32)
    b = rng.normal(size=(k, n)).astype(np.float32)
    t0 = time.perf_counter(); y = matmul_tiled_cpu(a, b, 32, 32, 32); t1 = time.perf_counter()
    assert np.allclose(y, a @ b, atol=1e-4)
    return {'cpu_seconds': t1 - t0, 'flops': float(2 * m * n * k)}


def smoke_test() -> None:
    rng = np.random.default_rng(4)
    a = rng.normal(size=(23, 31)).astype(np.float32)
    b = rng.normal(size=(31, 17)).astype(np.float32)
    y = matmul_tiled_cpu(a, b, 8, 7, 9)
    assert np.allclose(y, a @ b, atol=1e-4)

if __name__ == '__main__':
    print(benchmark_stub())
