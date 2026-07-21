from __future__ import annotations
"""Stage 13.6 — Decode bandwidth model.

Decode attention is often bandwidth-bound.

Per generated token, per layer, logical KV bytes:
    2 * num_heads * seq_len * head_dim * bytes_per_element

Theoretical tokens/sec upper bound from bandwidth:
    bandwidth_bytes_per_sec / bytes_per_token
"""


def decode_kv_bytes(num_heads: int, seq_len: int, head_dim: int, bytes_per_element: int = 2) -> int:
    return 2 * num_heads * seq_len * head_dim * bytes_per_element


def decode_tokens_per_second_upper_bound(bandwidth_gbs: float, num_heads: int, seq_len: int, head_dim: int, bytes_per_element: int = 2) -> float:
    return bandwidth_gbs * 1e9 / max(decode_kv_bytes(num_heads, seq_len, head_dim, bytes_per_element), 1)


def smoke_test() -> None:
    assert decode_kv_bytes(2, 4, 8, 2) == 256
    assert decode_tokens_per_second_upper_bound(1000, 2, 4, 8, 2) > 1e6
