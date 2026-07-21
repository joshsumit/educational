from __future__ import annotations
"""Stage 9.5 — FlashAttention interview notes."""


def flashattention_core_answer() -> str:
    return 'FlashAttention avoids materializing the full attention score/probability matrices by streaming K/V tiles and maintaining online softmax statistics.'


def online_softmax_answer() -> str:
    return 'Online softmax maintains a running max m and denominator l; when m increases, old l and accumulator are rescaled by exp(m_old - m_new).'


def why_materialized_attention_is_expensive() -> str:
    return 'Materialized attention stores O(T^2) scores and probabilities, causing large memory footprint and global-memory traffic.'


def flashattention_limitations_answer() -> str:
    return 'A teaching FlashAttention kernel omits production details such as warp specialization, advanced scheduling, dropout, backward pass, and hardware-specific tuning.'


def smoke_test() -> None:
    assert 'online softmax' in flashattention_core_answer()
    assert 'rescaled' in online_softmax_answer()
    assert 'O(T^2)' in why_materialized_attention_is_expensive()
