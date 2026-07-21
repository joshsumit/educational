from __future__ import annotations
"""Stage 12.3 — Triton INT8 matmul teaching kernel.

Inputs:
    A int8 [M,K]
    B int8 [K,N]

Compute:
    acc_i32 = A_i8 @ B_i8
    C_fp32  = acc_i32 * a_scale * b_scale

Teaching limitations:
    - per-tensor scales
    - fp32 output
    - no tensor-core-specific int8 path tuning
    - production kernels require deeper hardware-specific tuning
"""

import numpy as np

try:
    import torch
    import triton
    import triton.language as tl
except Exception:
    torch = triton = tl = None


def int8_matmul_reference_from_q(aq: np.ndarray, bq: np.ndarray, a_scale: float, b_scale: float) -> np.ndarray:
    return (aq.astype(np.int32) @ bq.astype(np.int32)).astype(np.float32) * a_scale * b_scale


if tl is not None:
    @triton.jit
    def int8_matmul_kernel(a_ptr, b_ptr, c_ptr,
                           M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                           stride_am: tl.constexpr, stride_ak: tl.constexpr,
                           stride_bk: tl.constexpr, stride_bn: tl.constexpr,
                           stride_cm: tl.constexpr, stride_cn: tl.constexpr,
                           a_scale: tl.constexpr, b_scale: tl.constexpr,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.int32)
        for k0 in range(0, K, BLOCK_K):
            a = tl.load(a_ptr + offs_m[:, None] * stride_am + (k0 + offs_k[None, :]) * stride_ak,
                        mask=(offs_m[:, None] < M) & ((k0 + offs_k[None, :]) < K), other=0)
            b = tl.load(b_ptr + (k0 + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn,
                        mask=((k0 + offs_k[:, None]) < K) & (offs_n[None, :] < N), other=0)
            acc += tl.dot(a, b, out_dtype=tl.int32)
        out = acc.to(tl.float32) * a_scale * b_scale
        tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
                 out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def int8_matmul_triton(aq, bq, a_scale: float, b_scale: float, block_m: int = 16, block_n: int = 16, block_k: int = 32):
    if triton is None or torch is None: raise RuntimeError('Triton/Torch not available')
    M, K = aq.shape; K2, N = bq.shape
    if K != K2: raise ValueError('shape mismatch')
    c = torch.empty((M, N), device=aq.device, dtype=torch.float32)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    int8_matmul_kernel[grid](aq, bq, c, M, N, K, aq.stride(0), aq.stride(1), bq.stride(0), bq.stride(1), c.stride(0), c.stride(1), a_scale, b_scale, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k)
    return c


def smoke_test() -> None:
    rng = np.random.default_rng(2)
    aq = rng.integers(-10, 10, size=(7, 9), dtype=np.int8)
    bq = rng.integers(-10, 10, size=(9, 5), dtype=np.int8)
    out = int8_matmul_reference_from_q(aq, bq, 0.1, 0.2)
    assert out.shape == (7, 5)

if __name__ == '__main__': smoke_test(); print('ok')
