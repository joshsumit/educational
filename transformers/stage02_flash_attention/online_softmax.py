"""
Online softmax: the mathematical core of FlashAttention.

Standard softmax over a vector x:
    p_i = exp(x_i - m) / sum_j exp(x_j - m)
    m = max_j x_j

Online softmax processes x in blocks and maintains:
    m = running maximum
    l = running sum of exp(x - m)

When a new block has a larger maximum, the old denominator must be rescaled.
"""
from __future__ import annotations
import torch


def online_softmax_1d(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Compute exact softmax using block-wise online normalization."""
    m = torch.tensor(float('-inf'), dtype=x.dtype, device=x.device)
    l = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    blocks = []

    for start in range(0, x.numel(), block_size):
        xb = x[start:start+block_size]
        mb = xb.max()
        m_new = torch.maximum(m, mb)
        # Old sum was measured around old m. Convert it to the new m_new frame.
        l = l * torch.exp(m - m_new) + torch.exp(xb - m_new).sum()
        m = m_new
        blocks.append(xb)

    return torch.exp(x - m) / l


def online_softmax_merge(m1: torch.Tensor, l1: torch.Tensor, m2: torch.Tensor, l2: torch.Tensor):
    """
    Merge two online-softmax states.

    This is important for parallel FlashAttention variants where different CTAs/warps own
    different key blocks and later combine partial statistics.
    """
    m = torch.maximum(m1, m2)
    l = l1 * torch.exp(m1 - m) + l2 * torch.exp(m2 - m)
    return m, l


def smoke_test() -> None:
    x = torch.randn(103)
    assert torch.allclose(online_softmax_1d(x, 17), torch.softmax(x, dim=0), atol=1e-6)
