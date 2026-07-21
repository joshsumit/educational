from __future__ import annotations
"""Stage 10.0 — Prefill vs decode.

Transformer inference has two very different phases.

Prefill:
    Input prompt has many tokens.
    Q has many rows.
    Attention looks like matrix/tiled attention.

Decode:
    Each active sequence usually contributes one new query token.
    Q has one row per sequence/head.
    The kernel streams all previous K/V cache entries.

Key consequence:
    Decode attention is often memory-bandwidth bound because it reads K and V for the whole context but does
    relatively little compute per byte compared with large prefill matmuls.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionPhaseEstimate:
    phase: str
    query_tokens: int
    key_tokens: int
    head_dim: int
    approximate_flops: int
    approximate_kv_bytes: int


def estimate_prefill(t: int, head_dim: int, bytes_per_element: int = 2) -> AttentionPhaseEstimate:
    flops = 4 * t * t * head_dim  # QK and PV, approximate multiply-add counts
    kv_bytes = 2 * t * head_dim * bytes_per_element
    return AttentionPhaseEstimate('prefill', t, t, head_dim, flops, kv_bytes)


def estimate_decode(seq_len: int, head_dim: int, bytes_per_element: int = 2) -> AttentionPhaseEstimate:
    flops = 4 * seq_len * head_dim  # one query: QK + PV
    kv_bytes = 2 * seq_len * head_dim * bytes_per_element
    return AttentionPhaseEstimate('decode', 1, seq_len, head_dim, flops, kv_bytes)


def decode_is_memory_bound_hint(seq_len: int, head_dim: int, bytes_per_element: int = 2) -> float:
    est = estimate_decode(seq_len, head_dim, bytes_per_element)
    return est.approximate_flops / max(est.approximate_kv_bytes, 1)


def smoke_test() -> None:
    p = estimate_prefill(128, 64)
    d = estimate_decode(128, 64)
    assert p.query_tokens == 128 and d.query_tokens == 1
    assert decode_is_memory_bound_hint(128, 64) == 1.0
