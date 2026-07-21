"""
RoPE and ALiBi implementations.

RoPE rotates even/odd channel pairs:
    [a, b] -> [a*cos(theta) - b*sin(theta), a*sin(theta) + b*cos(theta)]

ALiBi adds a head-specific linear bias to attention scores so nearby tokens receive preference.
"""
from __future__ import annotations
import math
import torch


def rope_cache(seq_len: int, head_dim: int, base: float = 10000.0, device=None):
    if head_dim % 2 != 0:
        raise ValueError('head_dim must be even for RoPE')
    pos = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    idx = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = torch.exp(-math.log(base) * idx / head_dim)
    theta = pos * inv_freq
    return torch.cos(theta), torch.sin(theta)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """Apply RoPE to [B,H,T,Dh]."""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    c = cos[offset:offset+x.shape[-2]].view(1, 1, x.shape[-2], -1).to(x.device, x.dtype)
    s = sin[offset:offset+x.shape[-2]].view(1, 1, x.shape[-2], -1).to(x.device, x.dtype)
    y_even = x_even * c - x_odd * s
    y_odd = x_even * s + x_odd * c
    y = torch.empty_like(x)
    y[..., 0::2] = y_even
    y[..., 1::2] = y_odd
    return y


def alibi_slopes(num_heads: int) -> torch.Tensor:
    """Simple monotonic slope schedule suitable for study/reference use."""
    h = torch.arange(1, num_heads + 1, dtype=torch.float32)
    return torch.pow(2.0, -8.0 * h / num_heads)


def alibi_bias(num_heads: int, q_len: int, k_len: int, device=None) -> torch.Tensor:
    q_pos = torch.arange(q_len, device=device).unsqueeze(1)
    k_pos = torch.arange(k_len, device=device).unsqueeze(0)
    distance = (k_pos - q_pos).abs().float()
    slopes = alibi_slopes(num_heads).to(device).view(num_heads, 1, 1)
    return -slopes * distance


def smoke_test() -> None:
    x = torch.randn(2, 4, 5, 8)
    cos, sin = rope_cache(5, 8)
    y = apply_rope(x, cos, sin)
    assert y.shape == x.shape
    assert alibi_bias(4, 5, 5).shape == (4, 5, 5)
