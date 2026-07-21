from __future__ import annotations
"""Stage 7.3 — Batched Triton matmul.

Program mapping:
    axis 0 / flattened pid: output tile over M,N
    axis 1: batch id

For each batch b:
    C[b] = A[b] @ B[b]

This prepares for attention where batch may mean:
    batch_id * num_heads + head_id

The kernel supports contiguous [B, M, K] and [B, K, N] tensors via explicit batch/matrix strides.
"""

import numpy as np

try:
    import torch
    import triton
    import triton.language as tl
except Exception:
    torch = triton = tl = None


def batched_matmul_cpu(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.matmul(a.astype(np.float32), b.astype(np.float32))


if tl is not None:
    @triton.jit
    def batched_matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        BATCH: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        stride_ab: tl.constexpr, stride_am: tl.constexpr, stride_ak: tl.constexpr,
        stride_bb: tl.constexpr, stride_bk: tl.constexpr, stride_bn: tl.constexpr,
        stride_cb: tl.constexpr, stride_cm: tl.constexpr, stride_cn: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_tile = tl.program_id(0)
        batch = tl.program_id(1)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        pid_m = pid_tile // num_pid_n
        pid_n = pid_tile % num_pid_n
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_base = a_ptr + batch * stride_ab
        b_base = b_ptr + batch * stride_bb
        c_base = c_ptr + batch * stride_cb
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k0 in range(0, K, BLOCK_K):
            a = tl.load(a_base + offs_m[:, None] * stride_am + (k0 + offs_k[None, :]) * stride_ak,
                        mask=(offs_m[:, None] < M) & ((k0 + offs_k[None, :]) < K), other=0.0)
            b = tl.load(b_base + (k0 + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn,
                        mask=((k0 + offs_k[:, None]) < K) & (offs_n[None, :] < N), other=0.0)
            acc += tl.dot(a, b)
        tl.store(c_base + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
                 acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def batched_matmul_triton(a, b, block_m: int = 16, block_n: int = 16, block_k: int = 32):
    if triton is None or torch is None:
        raise RuntimeError('Triton/Torch not available')
    BATCH, M, K = a.shape
    B2, K2, N = b.shape
    if BATCH != B2 or K != K2:
        raise ValueError('shape mismatch')
    c = torch.empty((BATCH, M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n), BATCH)
    batched_matmul_kernel[grid](a, b, c, BATCH, M, N, K,
                                a.stride(0), a.stride(1), a.stride(2),
                                b.stride(0), b.stride(1), b.stride(2),
                                c.stride(0), c.stride(1), c.stride(2),
                                BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k)
    return c


def smoke_test() -> None:
    rng = np.random.default_rng(8)
    a = rng.normal(size=(3, 11, 13)).astype(np.float32)
    b = rng.normal(size=(3, 13, 7)).astype(np.float32)
    assert np.allclose(batched_matmul_cpu(a, b), np.matmul(a, b), atol=1e-5)

if __name__ == '__main__':
    smoke_test(); print('ok')
