from __future__ import annotations
"""Stage 11.6 — Paged attention memory model.

Paged KV cache memory:

    num_blocks * block_size * num_heads * head_dim * 2(K/V) * bytes_per_element

Internal fragmentation:
    If a sequence uses N tokens and block_size is B, the last block may waste:

        ceil(N/B)*B - N

Paged allocation trades some fragmentation for dynamic allocation and better serving flexibility.
"""

import math


def paged_kv_cache_bytes(num_blocks: int, block_size: int, num_heads: int, head_dim: int, bytes_per_element: int = 2) -> int:
    return num_blocks * block_size * num_heads * head_dim * 2 * bytes_per_element


def blocks_needed(seq_len: int, block_size: int) -> int:
    return math.ceil(seq_len / block_size)


def internal_fragmentation_tokens(seq_len: int, block_size: int) -> int:
    return blocks_needed(seq_len, block_size) * block_size - seq_len


def total_fragmentation_tokens(seq_lens: list[int], block_size: int) -> int:
    return sum(internal_fragmentation_tokens(s, block_size) for s in seq_lens)


def smoke_test() -> None:
    assert blocks_needed(33, 16) == 3
    assert internal_fragmentation_tokens(33, 16) == 15
    assert paged_kv_cache_bytes(10, 16, 2, 8, 2) == 10240
    assert total_fragmentation_tokens([1, 16, 17], 16) == 30
