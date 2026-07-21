from __future__ import annotations
"""Stage 11.3 — Paged attention Triton skeleton.

This is a teaching skeleton for one query/head/request.

Inputs:
    q[D]
    k_cache[num_blocks, num_heads, block_size, D]
    v_cache[num_blocks, num_heads, block_size, DV]
    block_table[max_blocks_per_request]
    seq_len
    head_id

Algorithm:
    stream logical token blocks
    map logical block -> physical block using block_table
    load K/V vectors from physical block
    update online softmax stats
    accumulate output

Why skeleton, not final production kernel?
    Real paged attention kernels handle multiple sequences, flattened active tokens, vectorized heads, different
    head dims, quantized KV, block tables for batches, and carefully tuned memory access. This file teaches the
    core metadata path without hiding it behind excessive production complexity.
"""

import math
import numpy as np

try:
    import torch
    import triton
    import triton.language as tl
except Exception:
    torch = triton = tl = None


def paged_attention_cpu_skeleton(q: np.ndarray, k_cache: np.ndarray, v_cache: np.ndarray, block_table: np.ndarray, seq_len: int, head: int) -> np.ndarray:
    _, _, block_size, d = k_cache.shape
    dv = v_cache.shape[-1]
    m = -math.inf
    l = 0.0
    acc = np.zeros((dv,), dtype=np.float32)
    for pos in range(seq_len):
        logical = pos // block_size
        off = pos % block_size
        physical = int(block_table[logical])
        k = k_cache[physical, head, off].astype(np.float32)
        v = v_cache[physical, head, off].astype(np.float32)
        score = float(np.dot(q.astype(np.float32), k) / math.sqrt(d))
        m_new = max(m, score)
        old_scale = 0.0 if m == -math.inf else math.exp(m - m_new)
        p = math.exp(score - m_new)
        l = l * old_scale + p
        acc = acc * old_scale + p * v
        m = m_new
    return acc / l


if tl is not None:
    @triton.jit
    def paged_attention_kernel_skeleton(q_ptr, k_ptr, v_ptr, table_ptr, out_ptr,
                                        SEQ_LEN: tl.constexpr, NUM_HEADS: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                                        D: tl.constexpr, DV: tl.constexpr, HEAD_ID: tl.constexpr,
                                        stride_kb: tl.constexpr, stride_kh: tl.constexpr, stride_kt: tl.constexpr, stride_kd: tl.constexpr,
                                        stride_vb: tl.constexpr, stride_vh: tl.constexpr, stride_vt: tl.constexpr, stride_vd: tl.constexpr,
                                        scale: tl.constexpr,
                                        BLOCK_D: tl.constexpr, BLOCK_DV: tl.constexpr):
        # This skeleton processes one token at a time for clarity.
        # Production kernels process vectors of tokens/blocks per program.
        offs_d = tl.arange(0, BLOCK_D)
        offs_dv = tl.arange(0, BLOCK_DV)
        q = tl.load(q_ptr + offs_d, mask=offs_d < D, other=0.0).to(tl.float32)
        m = tl.full((), -float('inf'), tl.float32)
        l = tl.full((), 0.0, tl.float32)
        acc = tl.zeros((BLOCK_DV,), tl.float32)
        for pos in range(0, SEQ_LEN):
            logical = pos // BLOCK_SIZE
            off = pos - logical * BLOCK_SIZE
            physical = tl.load(table_ptr + logical)
            k = tl.load(k_ptr + physical * stride_kb + HEAD_ID * stride_kh + off * stride_kt + offs_d * stride_kd,
                        mask=offs_d < D, other=0.0).to(tl.float32)
            v = tl.load(v_ptr + physical * stride_vb + HEAD_ID * stride_vh + off * stride_vt + offs_dv * stride_vd,
                        mask=offs_dv < DV, other=0.0).to(tl.float32)
            score = tl.sum(q * k, axis=0) * scale
            m_new = tl.maximum(m, score)
            old_scale = tl.exp(m - m_new)
            old_scale = tl.where(m == -float('inf'), 0.0, old_scale)
            p = tl.exp(score - m_new)
            l = l * old_scale + p
            acc = acc * old_scale + p * v
            m = m_new
        out = acc / l
        tl.store(out_ptr + offs_dv, out, mask=offs_dv < DV)


def smoke_test() -> None:
    rng = np.random.default_rng(4)
    k_cache = rng.normal(size=(6, 2, 4, 8)).astype(np.float32)
    v_cache = rng.normal(size=(6, 2, 4, 8)).astype(np.float32)
    table = np.array([5, 2, 1], dtype=np.int32)
    q = rng.normal(size=(8,)).astype(np.float32)
    out = paged_attention_cpu_skeleton(q, k_cache, v_cache, table, seq_len=9, head=1)
    assert out.shape == (8,)
    assert np.all(np.isfinite(out))

if __name__ == '__main__':
    smoke_test(); print('ok')
