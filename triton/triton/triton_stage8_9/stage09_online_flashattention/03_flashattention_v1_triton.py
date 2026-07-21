from __future__ import annotations
"""Stage 9.3 — Teaching FlashAttention v1 Triton kernel.

This is a simplified single-head FlashAttention-style kernel.

Program mapping:
    one program computes a block of query rows [BLOCK_M] and all Dv output columns.

Algorithm inside one program:
    for each K/V block:
        scores = Q_block @ K_block.T / sqrt(D)
        apply causal mask
        update running row max m_i
        update running denominator l_i
        rescale old accumulator
        acc += exp(scores - m_new) @ V_block

At the end:
    out = acc / l_i

Teaching limitations:
    - Single head.
    - D and DV must fit BLOCK_D/BLOCK_DV.
    - Causal prefill case is the focus.
    - Real production kernels have more specialization and tuning.

Still, this captures the core idea: avoid materializing S and P.
"""

import math
import numpy as np

try:
    import torch
    import triton
    import triton.language as tl
except Exception:
    torch = triton = tl = None

import importlib
_ref = importlib.import_module('stage09_online_flashattention.02_flashattention_v1_reference')
flashattention_v1_reference = _ref.flashattention_v1_reference
attention_reference = _ref.attention_reference


if tl is not None:
    @triton.jit
    def flashattention_v1_kernel(q_ptr, k_ptr, v_ptr, o_ptr,
                                 T: tl.constexpr, D: tl.constexpr, DV: tl.constexpr,
                                 stride_qt: tl.constexpr, stride_qd: tl.constexpr,
                                 stride_kt: tl.constexpr, stride_kd: tl.constexpr,
                                 stride_vt: tl.constexpr, stride_vd: tl.constexpr,
                                 stride_ot: tl.constexpr, stride_od: tl.constexpr,
                                 scale: tl.constexpr,
                                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_DV: tl.constexpr):
        pid_m = tl.program_id(0)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        offs_dv = tl.arange(0, BLOCK_DV)

        q = tl.load(q_ptr + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd,
                    mask=(offs_m[:, None] < T) & (offs_d[None, :] < D), other=0.0)

        m_i = tl.full((BLOCK_M,), -float('inf'), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_DV), tl.float32)

        for n0 in range(0, T, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            k = tl.load(k_ptr + offs_n[None, :] * stride_kt + offs_d[:, None] * stride_kd,
                        mask=(offs_n[None, :] < T) & (offs_d[:, None] < D), other=0.0)
            v = tl.load(v_ptr + offs_n[:, None] * stride_vt + offs_dv[None, :] * stride_vd,
                        mask=(offs_n[:, None] < T) & (offs_dv[None, :] < DV), other=0.0)

            scores = tl.dot(q, k) * scale
            causal = offs_n[None, :] <= offs_m[:, None]
            valid = (offs_m[:, None] < T) & (offs_n[None, :] < T) & causal
            scores = tl.where(valid, scores, -float('inf'))

            m_block = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_block)
            old_scale = tl.exp(m_i - m_new)
            old_scale = tl.where(m_i == -float('inf'), 0.0, old_scale)
            p = tl.exp(scores - m_new[:, None])
            p = tl.where(valid, p, 0.0)
            l_new = l_i * old_scale + tl.sum(p, axis=1)
            acc = acc * old_scale[:, None] + tl.dot(p, v)
            m_i = m_new
            l_i = l_new

        out = acc / l_i[:, None]
        tl.store(o_ptr + offs_m[:, None] * stride_ot + offs_dv[None, :] * stride_od,
                 out, mask=(offs_m[:, None] < T) & (offs_dv[None, :] < DV))


def flashattention_v1_triton(q, k, v, block_m: int = 16, block_n: int = 32, block_d: int | None = None, block_dv: int | None = None):
    if triton is None or torch is None:
        raise RuntimeError('Triton/Torch not available')
    T, D = q.shape
    T2, D2 = k.shape
    T3, DV = v.shape
    if T != T2 or T != T3 or D != D2:
        raise ValueError('shape mismatch')
    block_d = block_d or (1 << (D - 1).bit_length())
    block_dv = block_dv or (1 << (DV - 1).bit_length())
    out = torch.empty((T, DV), device=q.device, dtype=torch.float32)
    grid = (triton.cdiv(T, block_m),)
    flashattention_v1_kernel[grid](q, k, v, out, T, D, DV, q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1), out.stride(0), out.stride(1), 1.0 / math.sqrt(D), BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=block_d, BLOCK_DV=block_dv)
    return out


def smoke_test() -> None:
    rng = np.random.default_rng(7)
    q = rng.normal(size=(17, 8)).astype(np.float32)
    k = rng.normal(size=(17, 8)).astype(np.float32)
    v = rng.normal(size=(17, 8)).astype(np.float32)
    out = flashattention_v1_reference(q, k, v, block_m=4, block_n=5, causal=True)
    exp = attention_reference(q, k, v, causal=True)
    assert np.allclose(out, exp, atol=1e-5)

if __name__ == '__main__':
    smoke_test(); print('ok')
