from __future__ import annotations
"""Stage 8.4 — Triton QK^T score kernel.

Goal:
    Compute scores = Q @ K.T / sqrt(D)

Program mapping:
    One program computes one tile of scores [BLOCK_M, BLOCK_N].

This is almost the tiled matmul kernel from Stage 6, except B is K transposed logically:

    K is stored [Tk, D]
    Load K tile as [BLOCK_N, BLOCK_D]
    Use tl.trans(k_tile) inside tl.dot or load it in transposed tile shape.

This kernel only computes scores. It does not apply softmax or multiply by V.
"""

import math
import numpy as np

try:
    import torch
    import triton
    import triton.language as tl
except Exception:
    torch = triton = tl = None


def qk_reference(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) @ k.astype(np.float32).T / math.sqrt(q.shape[1])


if tl is not None:
    @triton.jit
    def qk_kernel(q_ptr, k_ptr, s_ptr,
                  TQ: tl.constexpr, TK: tl.constexpr, D: tl.constexpr,
                  stride_qt: tl.constexpr, stride_qd: tl.constexpr,
                  stride_kt: tl.constexpr, stride_kd: tl.constexpr,
                  stride_st: tl.constexpr, stride_sk: tl.constexpr,
                  scale: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for d0 in range(0, D, BLOCK_D):
            q = tl.load(q_ptr + offs_m[:, None] * stride_qt + (d0 + offs_d[None, :]) * stride_qd,
                        mask=(offs_m[:, None] < TQ) & ((d0 + offs_d[None, :]) < D), other=0.0)
            k = tl.load(k_ptr + offs_n[None, :] * stride_kt + (d0 + offs_d[:, None]) * stride_kd,
                        mask=(offs_n[None, :] < TK) & ((d0 + offs_d[:, None]) < D), other=0.0)
            acc += tl.dot(q, k)
        acc = acc * scale
        tl.store(s_ptr + offs_m[:, None] * stride_st + offs_n[None, :] * stride_sk,
                 acc, mask=(offs_m[:, None] < TQ) & (offs_n[None, :] < TK))


def qk_triton(q, k, block_m: int = 16, block_n: int = 16, block_d: int = 32):
    if triton is None or torch is None:
        raise RuntimeError('Triton/Torch not available')
    TQ, D = q.shape
    TK, D2 = k.shape
    if D != D2:
        raise ValueError('shape mismatch')
    scores = torch.empty((TQ, TK), device=q.device, dtype=torch.float32)
    grid = (triton.cdiv(TQ, block_m), triton.cdiv(TK, block_n))
    qk_kernel[grid](q, k, scores, TQ, TK, D, q.stride(0), q.stride(1), k.stride(0), k.stride(1), scores.stride(0), scores.stride(1), 1.0 / math.sqrt(D), BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=block_d)
    return scores


def smoke_test() -> None:
    rng = np.random.default_rng(2)
    q = rng.normal(size=(9, 13)).astype(np.float32)
    k = rng.normal(size=(7, 13)).astype(np.float32)
    assert np.allclose(qk_reference(q, k), q @ k.T / math.sqrt(13), atol=1e-5)

if __name__ == '__main__':
    smoke_test(); print('ok')
