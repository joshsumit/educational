from __future__ import annotations
"""Stage 10.5 — KV cache memory model.

For each layer, KV cache memory is approximately:

    2 * batch * num_heads * seq_len * head_dim * bytes_per_element

The factor 2 is for K and V.

For decode, each new token often streams K and V for the active sequence window. This is why decode attention
frequently becomes memory-bandwidth bound.
"""


def kv_cache_bytes(batch: int, num_heads: int, seq_len: int, head_dim: int, bytes_per_element: int = 2, num_layers: int = 1) -> int:
    return 2 * batch * num_layers * num_heads * seq_len * head_dim * bytes_per_element


def decode_kv_bytes_per_token(num_heads: int, seq_len: int, head_dim: int, bytes_per_element: int = 2) -> int:
    return 2 * num_heads * seq_len * head_dim * bytes_per_element


def max_tokens_for_budget(memory_bytes: int, batch: int, num_heads: int, head_dim: int, bytes_per_element: int = 2, num_layers: int = 1) -> int:
    denom = 2 * batch * num_layers * num_heads * head_dim * bytes_per_element
    return memory_bytes // denom


def smoke_test() -> None:
    assert kv_cache_bytes(1, 2, 4, 8, 2, 1) == 256
    assert decode_kv_bytes_per_token(2, 4, 8, 2) == 256
    assert max_tokens_for_budget(256, 1, 2, 8, 2, 1) == 4
