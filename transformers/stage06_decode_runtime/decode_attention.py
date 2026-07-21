"""
Prefill vs decode attention.

Prefill:
    Q length = prompt length T
    K length = prompt length T
    Cost O(T^2 * Dh)

Decode:
    Q length = 1
    K length = current history T
    Cost O(T * Dh)
    Usually memory-bandwidth bound because every step streams K/V history.
"""
from __future__ import annotations
import math
import torch


def decode_attention_one_token(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, num_q_heads: int | None = None) -> torch.Tensor:
    """
    Args:
        q:       [Hq,Dh]
        k_cache: [Hkv,T,Dh]
        v_cache: [Hkv,T,Dh]

    Supports MHA, MQA, and GQA by repeating KV heads to Hq.
    """
    hq, dh = q.shape
    hkv = k_cache.shape[0]
    if hq % hkv != 0:
        raise ValueError('Hq must be divisible by Hkv for MQA/GQA decode')
    repeat = hq // hkv
    k = k_cache.repeat_interleave(repeat, dim=0)
    v = v_cache.repeat_interleave(repeat, dim=0)
    scores = torch.einsum('hd,htd->ht', q, k) / math.sqrt(dh)
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum('ht,htd->hd', probs, v)
    return out


def decode_bytes_read(seq_len: int, num_kv_heads: int, head_dim: int, bytes_per_elem: int = 2) -> int:
    """Approximate bytes streamed for K and V during a one-token decode step."""
    return 2 * seq_len * num_kv_heads * head_dim * bytes_per_elem


def smoke_test() -> None:
    q = torch.randn(4, 8)
    k = torch.randn(2, 5, 8)
    v = torch.randn(2, 5, 8)
    out = decode_attention_one_token(q, k, v)
    assert out.shape == q.shape
    assert decode_bytes_read(10, 2, 8, 2) == 640
