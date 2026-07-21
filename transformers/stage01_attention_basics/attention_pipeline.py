"""
Full scaled dot-product attention pipeline.

Pipeline:
    1. QK^T matmul
    2. scale by 1/sqrt(Dh)
    3. add mask or causal bias
    4. softmax over keys
    5. multiply probabilities by V

This file intentionally materializes scores and probabilities, so it is easy to compare with
FlashAttention later.
"""
from __future__ import annotations
import math
import torch


def causal_mask(tq: int, tk: int, device=None) -> torch.Tensor:
    """Return a [Tq,Tk] mask where True means the position is allowed."""
    # For self-attention with tq == tk, query i can see keys <= i.
    q_pos = torch.arange(tq, device=device).unsqueeze(1)
    k_pos = torch.arange(tk, device=device).unsqueeze(0)
    # Handles decode too: if tq=1 and tk=history_len, caller usually does not use this mask.
    return k_pos <= q_pos + (tk - tq)


def scaled_dot_product_attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False):
    """
    Args:
        q: [B,H,Tq,Dh]
        k: [B,H,Tk,Dh]
        v: [B,H,Tk,Dv]

    Returns:
        out:   [B,H,Tq,Dv]
        probs: [B,H,Tq,Tk]
    """
    dh = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(dh))
    if causal:
        allowed = causal_mask(q.shape[-2], k.shape[-2], device=q.device)
        scores = scores.masked_fill(~allowed, float('-inf'))
    # softmax subtracts max internally in PyTorch, but the conceptual step is normalize exp(scores).
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)
    return out, probs


def multi_head_attention(q_btd: torch.Tensor, k_btd: torch.Tensor, v_btd: torch.Tensor, num_heads: int, causal: bool = False) -> torch.Tensor:
    """MHA wrapper from [B,T,D] to [B,T,D]."""
    b, tq, d = q_btd.shape
    dh = d // num_heads
    q = q_btd.view(b, tq, num_heads, dh).transpose(1, 2)
    k = k_btd.view(b, k_btd.shape[1], num_heads, dh).transpose(1, 2)
    v = v_btd.view(b, v_btd.shape[1], num_heads, dh).transpose(1, 2)
    out, _ = scaled_dot_product_attention_reference(q, k, v, causal=causal)
    return out.transpose(1, 2).contiguous().view(b, tq, d)


def multi_query_attention(q_btd: torch.Tensor, k_btdh: torch.Tensor, v_btdh: torch.Tensor, num_q_heads: int, causal: bool = False) -> torch.Tensor:
    """
    MQA: all query heads share one K/V head.

    q_btd:  [B,T,Dq]
    k_btdh: [B,T,Dh]
    v_btdh: [B,T,Dh]

    Main benefit:
        KV cache size is reduced by approximately num_q_heads compared with MHA.
    """
    b, tq, dq = q_btd.shape
    dh = dq // num_q_heads
    q = q_btd.view(b, tq, num_q_heads, dh).transpose(1, 2)
    k = k_btdh.unsqueeze(1).expand(b, num_q_heads, k_btdh.shape[1], dh)
    v = v_btdh.unsqueeze(1).expand(b, num_q_heads, v_btdh.shape[1], dh)
    out, _ = scaled_dot_product_attention_reference(q, k, v, causal=causal)
    return out.transpose(1, 2).contiguous().view(b, tq, dq)


def grouped_query_attention(q_btd: torch.Tensor, k_btkd: torch.Tensor, v_btkd: torch.Tensor, num_q_heads: int, num_kv_heads: int, causal: bool = False) -> torch.Tensor:
    """
    GQA: groups of query heads share each K/V head.

    Example:
        num_q_heads=32, num_kv_heads=8 -> every KV head serves 4 query heads.
    """
    if num_q_heads % num_kv_heads != 0:
        raise ValueError('num_q_heads must be divisible by num_kv_heads')
    b, tq, dq = q_btd.shape
    dh = dq // num_q_heads
    q = q_btd.view(b, tq, num_q_heads, dh).transpose(1, 2)
    k = k_btkd.transpose(1, 2)  # [B,Hkv,T,Dh]
    v = v_btkd.transpose(1, 2)
    repeat = num_q_heads // num_kv_heads
    k = k.repeat_interleave(repeat, dim=1)
    v = v.repeat_interleave(repeat, dim=1)
    out, _ = scaled_dot_product_attention_reference(q, k, v, causal=causal)
    return out.transpose(1, 2).contiguous().view(b, tq, dq)


def smoke_test() -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 4, 16)
    out = multi_head_attention(q, q, q, 4, causal=True)
    assert out.shape == q.shape
    k = torch.randn(2, 4, 4)
    v = torch.randn(2, 4, 4)
    assert multi_query_attention(q, k, v, 4).shape == q.shape
    kg = torch.randn(2, 4, 2, 4)
    vg = torch.randn(2, 4, 2, 4)
    assert grouped_query_attention(q, kg, vg, 4, 2).shape == q.shape
