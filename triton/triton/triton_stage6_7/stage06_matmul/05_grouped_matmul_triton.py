from __future__ import annotations
"""Stage 6.5 — Grouped program ordering for Triton matmul.

A high-performance Triton matmul often uses a flattened 1D program id and manually maps it into (pid_m, pid_n).
This allows grouped ordering.

Why grouped ordering exists:
    Suppose many neighboring M tiles use the same B tile. Scheduling those nearby programs close together can
    improve L2 cache reuse of B.

Naive row-major order:
    (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)

Grouped order with GROUP_M=2:
    (0,0), (1,0), (0,1), (1,1), (0,2), (1,2)

This file includes:
    - CPU grouped order simulation
    - CPU grouped tiled matmul
    - real Triton grouped matmul kernel
"""

import numpy as np

try:
    import torch
    import triton
    import triton.language as tl
except Exception:
    torch = triton = tl = None


def grouped_order(num_m: int, num_n: int, group_m: int) -> list[tuple[int, int]]:
    order: list[tuple[int, int]] = []
    for group_start_m in range(0, num_m, group_m):
        actual = min(group_m, num_m - group_start_m)
        for n in range(num_n):
            for local_m in range(actual):
                order.append((group_start_m + local_m, n))
    return order


def matmul_grouped_cpu(a: np.ndarray, b: np.ndarray, block_m: int = 16, block_n: int = 16, block_k: int = 32, group_m: int = 4) -> np.ndarray:
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError('shape mismatch')
    c = np.zeros((m, n), dtype=np.float32)
    num_m = (m + block_m - 1) // block_m
    num_n = (n + block_n - 1) // block_n
    for pid_m, pid_n in grouped_order(num_m, num_n, group_m):
        m0 = pid_m * block_m; n0 = pid_n * block_n
        m1 = min(m0 + block_m, m); n1 = min(n0 + block_n, n)
        acc = np.zeros((m1 - m0, n1 - n0), dtype=np.float32)
        for k0 in range(0, k, block_k):
            k1 = min(k0 + block_k, k)
            acc += a[m0:m1, k0:k1].astype(np.float32) @ b[k0:k1, n0:n1].astype(np.float32)
        c[m0:m1, n0:n1] = acc
    return c


if tl is not None:
    @triton.jit
    def matmul_grouped_kernel(
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


def matmul_grouped_triton(a, b, block_m: int = 32, block_n: int = 32, block_k: int = 32, group_m: int = 4):
    if triton is None or torch is None:
        raise RuntimeError('Triton/Torch not available')
    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise ValueError('shape mismatch')
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    matmul_grouped_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=group_m)
    return c


def smoke_test() -> None:
    assert grouped_order(4, 2, 2) == [(0,0),(1,0),(0,1),(1,1),(2,0),(3,0),(2,1),(3,1)]
    rng = np.random.default_rng(5)
    a = rng.normal(size=(19, 23)).astype(np.float32)
    b = rng.normal(size=(23, 13)).astype(np.float32)
    y = matmul_grouped_cpu(a, b, 6, 5, 7, 2)
    assert np.allclose(y, a @ b, atol=1e-4)

if __name__ == '__main__':
    smoke_test(); print('ok')
