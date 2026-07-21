"""
FlashAttention V2 partitioning simulator.

V1 lesson:
    One CTA often owns a Q block and streams over K blocks.

V2 lesson:
    Improve parallelism and reduce non-matmul work by changing work partitioning.
    This file simulates splitting K/V work into partitions and merging online-softmax states.
"""
from __future__ import annotations
import math
import torch
from .online_softmax import online_softmax_merge


def attention_partition_state(q: torch.Tensor, k_part: torch.Tensor, v_part: torch.Tensor):
    """Return partial online-softmax state for one query vector and one K/V partition."""
    scores = (k_part @ q) / math.sqrt(q.numel())
    m = scores.max()
    p_un = torch.exp(scores - m)
    l = p_un.sum()
    numerator = p_un @ v_part
    return m, l, numerator


def merge_partition_outputs(states):
    """Merge [(m,l,num), ...] into final attention output for one query row."""
    m, l, num = states[0]
    for m2, l2, num2 in states[1:]:
        m_new, l_new = online_softmax_merge(m, l, m2, l2)
        num = num * torch.exp(m - m_new) + num2 * torch.exp(m2 - m_new)
        m, l = m_new, l_new
    return num / l


def partitioned_attention_one_row(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, partitions: int = 2) -> torch.Tensor:
    chunks_k = torch.chunk(k, partitions, dim=0)
    chunks_v = torch.chunk(v, partitions, dim=0)
    states = [attention_partition_state(q, kp, vp) for kp, vp in zip(chunks_k, chunks_v)]
    return merge_partition_outputs(states)


def smoke_test() -> None:
    q = torch.randn(8)
    k = torch.randn(17, 8)
    v = torch.randn(17, 8)
    ref = torch.softmax((k @ q) / math.sqrt(8), dim=0) @ v
    got = partitioned_attention_one_row(q, k, v, 4)
    assert torch.allclose(got, ref, atol=1e-6)
