"""
FlashAttention reference implementation.

This computes exact attention without storing the full [Tq,Tk] attention matrix.
It is still Python/PyTorch, not CUDA, but the dataflow mirrors the kernel algorithm.

For each Q block:
    keep output accumulator O
    keep running row max m
    keep running row denominator l
    stream over K/V blocks
    update O using online softmax state transitions
"""
from __future__ import annotations
import math
import torch


def flash_attention_single_head(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_m: int = 64, block_n: int = 64, causal: bool = False) -> torch.Tensor:
    """
    Args:
        q: [Tq,Dh]
        k: [Tk,Dh]
        v: [Tk,Dv]
    Returns:
        out: [Tq,Dv]
    """
    tq, dh = q.shape
    tk = k.shape[0]
    dv = v.shape[1]
    scale = 1.0 / math.sqrt(dh)
    out = torch.empty((tq, dv), dtype=q.dtype, device=q.device)

    for m0 in range(0, tq, block_m):
        q_blk = q[m0:m0+block_m]
        rows = q_blk.shape[0]
        # m_i and l_i exist per query row.
        m_i = torch.full((rows,), float('-inf'), dtype=q.dtype, device=q.device)
        l_i = torch.zeros((rows,), dtype=q.dtype, device=q.device)
        acc = torch.zeros((rows, dv), dtype=q.dtype, device=q.device)

        for n0 in range(0, tk, block_n):
            k_blk = k[n0:n0+block_n]
            v_blk = v[n0:n0+block_n]
            scores = (q_blk @ k_blk.T) * scale
            if causal:
                q_pos = torch.arange(m0, m0 + rows, device=q.device).unsqueeze(1)
                k_pos = torch.arange(n0, n0 + k_blk.shape[0], device=q.device).unsqueeze(0)
                scores = scores.masked_fill(k_pos > q_pos, float('-inf'))

            block_m_i = scores.max(dim=1).values
            m_new = torch.maximum(m_i, block_m_i)

            # Rescale old accumulator because denominator frame changed from m_i to m_new.
            old_scale = torch.exp(m_i - m_new)
            p = torch.exp(scores - m_new.unsqueeze(1))
            l_new = l_i * old_scale + p.sum(dim=1)

            # Numerator update follows the same rescaling as denominator.
            acc = acc * old_scale.unsqueeze(1) + p @ v_blk
            m_i = m_new
            l_i = l_new

        out[m0:m0+rows] = acc / l_i.unsqueeze(1)
    return out


def flash_attention_batched(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_m: int = 64, block_n: int = 64, causal: bool = False) -> torch.Tensor:
    """Apply single-head FlashAttention over [B,H,T,D]."""
    bsz, heads, tq, _ = q.shape
    outs = []
    for b in range(bsz):
        per_h = []
        for h in range(heads):
            per_h.append(flash_attention_single_head(q[b,h], k[b,h], v[b,h], block_m, block_n, causal))
        outs.append(torch.stack(per_h, dim=0))
    return torch.stack(outs, dim=0)


def smoke_test() -> None:
    torch.manual_seed(0)
    q = torch.randn(1, 2, 9, 8)
    k = torch.randn(1, 2, 9, 8)
    v = torch.randn(1, 2, 9, 8)
    ref = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(8), dim=-1) @ v
    got = flash_attention_batched(q, k, v, 4, 5)
    assert torch.allclose(got, ref, atol=1e-5)
