"""
QK^T matmul implementations.

This file isolates the first GEMM in attention:
    S[b,h,i,j] = dot(Q[b,h,i,:], K[b,h,j,:])

Complexity:
    FLOPs approximately 2 * B * H * Tq * Tk * Dh
Memory output:
    B * H * Tq * Tk elements if materialized

Low-level interview point:
    Standard attention materializes the score matrix. FlashAttention avoids materializing it.
"""
from __future__ import annotations
import torch


def qk_matmul_naive(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Four-loop educational reference for [B,H,Tq,Dh] x [B,H,Tk,Dh]."""
    bsz, heads, tq, dh = q.shape
    _, _, tk, dh2 = k.shape
    if dh != dh2:
        raise ValueError('Q and K head dimensions must match')
    scores = torch.empty((bsz, heads, tq, tk), dtype=q.dtype, device=q.device)
    for b in range(bsz):
        for h in range(heads):
            for i in range(tq):
                acc = []
                for j in range(tk):
                    # In CUDA, this dot product would be distributed across lanes or tensor cores.
                    acc.append(torch.sum(q[b, h, i] * k[b, h, j]))
                scores[b, h, i] = torch.stack(acc)
    return scores


def qk_matmul_vectorized(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Production-like reference using batched matmul."""
    return torch.matmul(q, k.transpose(-1, -2))


def qk_matmul_tiled_reference(q: torch.Tensor, k: torch.Tensor, block_m: int = 16, block_n: int = 16) -> torch.Tensor:
    """
    Tiled simulation of a CTA-level QK kernel.

    CUDA mapping intuition:
        - A CTA owns a [block_m x block_n] tile of the score matrix.
        - Q tile: [block_m, Dh]
        - K tile: [block_n, Dh]
        - Dot products accumulate across Dh.

    This Python implementation uses torch.matmul inside each tile. The point is to make
    ownership and memory movement explicit, not to outperform torch.matmul.
    """
    bsz, heads, tq, dh = q.shape
    tk = k.shape[2]
    out = torch.empty((bsz, heads, tq, tk), dtype=q.dtype, device=q.device)
    for b in range(bsz):
        for h in range(heads):
            for m0 in range(0, tq, block_m):
                for n0 in range(0, tk, block_n):
                    q_tile = q[b, h, m0:m0+block_m, :]
                    k_tile = k[b, h, n0:n0+block_n, :]
                    out[b, h, m0:m0+q_tile.shape[0], n0:n0+k_tile.shape[0]] = q_tile @ k_tile.T
    return out


def smoke_test() -> None:
    q = torch.randn(1, 2, 5, 8)
    k = torch.randn(1, 2, 7, 8)
    ref = qk_matmul_vectorized(q, k)
    assert torch.allclose(qk_matmul_naive(q, k), ref, atol=1e-5)
    assert torch.allclose(qk_matmul_tiled_reference(q, k, 2, 3), ref, atol=1e-5)
