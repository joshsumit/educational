from __future__ import annotations
"""Stage 10.6 — Decode attention interview notes."""


def prefill_vs_decode_answer() -> str:
    return 'Prefill processes many query tokens and is matmul-heavy; decode processes one new query per sequence and streams the KV cache, so it is often memory-bandwidth bound.'


def decode_kernel_answer() -> str:
    return 'A decode attention kernel streams K/V blocks, computes QK scores, maintains online softmax max and denominator, and accumulates probability-weighted V.'


def kv_cache_answer() -> str:
    return 'KV cache stores previous keys and values per layer/head/token so decode does not recompute attention projections for past tokens.'


def smoke_test() -> None:
    assert 'memory-bandwidth' in prefill_vs_decode_answer()
    assert 'online softmax' in decode_kernel_answer()
    assert 'previous keys' in kv_cache_answer()
