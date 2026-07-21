from __future__ import annotations
"""Stage 7.1 — Autotuned Triton matmul.

This file shows the production-style idea:

    @triton.autotune(configs=[...], key=['M', 'N', 'K'])

Triton will benchmark candidate configurations and cache/select a good one for the shape key.

Important:
    CPU smoke tests do not require Triton. The real autotuned kernel is guarded behind optional imports.
"""

import numpy as np

try:
    import torch
    import triton
    import triton.language as tl
except Exception:
    torch = triton = tl = None


def matmul_reference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a.astype(np.float32) @ b.astype(np.float32)


if tl is not None:
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 32, 'GROUP_M': 4}, num_warps=4, num_stages=3),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 4}, num_warps=4, num_stages=3),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 4}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 4}, num_warps=8, num_stages=4),
        ],
        key=['M', 'N', 'K'],
    )
    @triton.jit
    def matmul_autotuned_kernel(
        a_ptr, b_ptr, c_ptr,
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        stride_am: tl.constexpr, stride_ak: tl.constexpr,
        stride_bk: tl.constexpr, stride_bn: tl.constexpr,
        stride_cm: tl.constexpr, stride_cn: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        local = pid % num_pid_in_group
        pid_m = first_pid_m + (local % group_size_m)
        pid_n = local // group_size_m
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k0 in range(0, K, BLOCK_K):
            a = tl.load(a_ptr + offs_m[:, None] * stride_am + (k0 + offs_k[None, :]) * stride_ak,
                        mask=(offs_m[:, None] < M) & ((k0 + offs_k[None, :]) < K), other=0.0)
            b = tl.load(b_ptr + (k0 + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn,
                        mask=((k0 + offs_k[:, None]) < K) & (offs_n[None, :] < N), other=0.0)
            acc += tl.dot(a, b)
        tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
                 acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul_autotuned_triton(a, b):
    if triton is None or torch is None:
        raise RuntimeError('Triton/Torch not available')
    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise ValueError('shape mismatch')
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    matmul_autotuned_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
    return c


def smoke_test() -> None:
    rng = np.random.default_rng(6)
    a = rng.normal(size=(9, 13)).astype(np.float32)
    b = rng.normal(size=(13, 7)).astype(np.float32)
    assert np.allclose(matmul_reference(a, b), a @ b, atol=1e-5)

if __name__ == '__main__':
    smoke_test(); print('ok')
