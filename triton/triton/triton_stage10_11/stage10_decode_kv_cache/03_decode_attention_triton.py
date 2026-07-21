from __future__ import annotations
"""Stage 10.3 — Triton decode attention kernel.

Teaching kernel:
    one program computes decode attention for one query/head.

Inputs:
    q[D]
    k_cache[S,D]
    v_cache[S,DV]

Algorithm:
    stream K/V in BLOCK_N chunks
    maintain online softmax stats m and l
    maintain output accumulator acc[DV]

Limitations:
    - single query/head
    - D and DV must fit BLOCK_D/BLOCK_DV
    - production kernels handle batches, multiple heads, pages, quantization, and scheduling metadata
"""

import math
import numpy as np

try:
    import torch
    import triton
    import triton.language as tl
except Exception:
    torch = triton = tl = None


def decode_attention_reference(q: np.ndarray, k_cache: np.ndarray, v_cache: np.ndarray) -> np.ndarray:
    scores = k_cache.astype(np.float32) @ q.astype(np.float32) / math.sqrt(q.size)
    p = np.exp(scores - np.max(scores)); p = p / np.sum(p)
    return p @ v_cache.astype(np.float32)


if tl is not None:
    @triton.jit
    def decode_attention_kernel(q_ptr, k_ptr, v_ptr, out_ptr,
                                S: tl.constexpr, D: tl.constexpr, DV: tl.constexpr,
                                stride_ks: tl.constexpr, stride_kd: tl.constexpr,
                                stride_vs: tl.constexpr, stride_vd: tl.constexpr,
                                scale: tl.constexpr,
                                BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_DV: tl.constexpr):
        offs_d = tl.arange(0, BLOCK_D)
        offs_dv = tl.arange(0, BLOCK_DV)
        q = tl.load(q_ptr + offs_d, mask=offs_d < D, other=0.0).to(tl.float32)
        m = tl.full((), -float('inf'), tl.float32)
        l = tl.full((), 0.0, tl.float32)
        acc = tl.zeros((BLOCK_DV,), tl.float32)
        for n0 in range(0, S, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            k = tl.load(k_ptr + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
                        mask=(offs_n[:, None] < S) & (offs_d[None, :] < D), other=0.0)
            v = tl.load(v_ptr + offs_n[:, None] * stride_vs + offs_dv[None, :] * stride_vd,
                        mask=(offs_n[:, None] < S) & (offs_dv[None, :] < DV), other=0.0)
            scores = tl.sum(k * q[None, :], axis=1) * scale
            scores = tl.where(offs_n < S, scores, -float('inf'))
            mb = tl.max(scores, axis=0)
            m_new = tl.maximum(m, mb)
            old_scale = tl.exp(m - m_new)
            old_scale = tl.where(m == -float('inf'), 0.0, old_scale)
            p = tl.exp(scores - m_new)
            p = tl.where(offs_n < S, p, 0.0)
            l = l * old_scale + tl.sum(p, axis=0)
            acc = acc * old_scale + tl.sum(p[:, None] * v, axis=0)
            m = m_new
        out = acc / l
        tl.store(out_ptr + offs_dv, out, mask=offs_dv < DV)


def decode_attention_triton(q, k_cache, v_cache, block_n: int = 128, block_d: int | None = None, block_dv: int | None = None):
    if triton is None or torch is None:
        raise RuntimeError('Triton/Torch not available')
    S, D = k_cache.shape
    S2, DV = v_cache.shape
    if q.numel() != D or S != S2:
        raise ValueError('shape mismatch')
    block_d = block_d or (1 << (D - 1).bit_length())
    block_dv = block_dv or (1 << (DV - 1).bit_length())
    out = torch.empty((DV,), device=q.device, dtype=torch.float32)
    decode_attention_kernel[(1,)](q, k_cache, v_cache, out, S, D, DV, k_cache.stride(0), k_cache.stride(1), v_cache.stride(0), v_cache.stride(1), 1.0 / math.sqrt(D), BLOCK_N=block_n, BLOCK_D=block_d, BLOCK_DV=block_dv)
    return out


def smoke_test() -> None:
    rng = np.random.default_rng(1)
    q = rng.normal(size=(8,)).astype(np.float32)
    k = rng.normal(size=(33, 8)).astype(np.float32)
    v = rng.normal(size=(33, 8)).astype(np.float32)
    out = decode_attention_reference(q, k, v)
    assert out.shape == (8,)

if __name__ == '__main__':
    smoke_test(); print('ok')
