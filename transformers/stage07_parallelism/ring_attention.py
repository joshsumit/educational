"""
Ring attention simulator for context parallelism.

When sequence length is too large for one device, split K/V context across ranks.
Each rank owns a block of tokens. Query blocks circulate or K/V blocks circulate in a ring.
The outputs are merged using online softmax state, not by concatenating full score matrices.
"""
from __future__ import annotations
import math
import torch
from stage02_flash_attention.online_softmax import online_softmax_merge


def ring_attention_one_query(q: torch.Tensor, k_shards: list[torch.Tensor], v_shards: list[torch.Tensor]) -> torch.Tensor:
    states = []
    for k, v in zip(k_shards, v_shards):
        scores = (k @ q) / math.sqrt(q.numel())
        m = scores.max()
        p = torch.exp(scores - m)
        l = p.sum()
        num = p @ v
        states.append((m, l, num))
    m, l, num = states[0]
    for m2, l2, num2 in states[1:]:
        m_new, l_new = online_softmax_merge(m, l, m2, l2)
        num = num * torch.exp(m - m_new) + num2 * torch.exp(m2 - m_new)
        m, l = m_new, l_new
    return num / l


def smoke_test() -> None:
    q = torch.randn(8)
    k = torch.randn(12,8)
    v = torch.randn(12,8)
    ref = torch.softmax((k @ q) / math.sqrt(8), dim=0) @ v
    got = ring_attention_one_query(q, list(torch.chunk(k,3)), list(torch.chunk(v,3)))
    assert torch.allclose(got, ref, atol=1e-6)
