from __future__ import annotations
"""Stage 11.2 — Paged attention reference.

This file uses a block table to gather logical K/V sequence data, then runs ordinary decode attention.

It is intentionally CPU-simple. The point is to validate the metadata path:

    request_id + position -> logical block -> physical block -> K/V vector
"""

import math
import numpy as np


def gather_paged_sequence(cache: np.ndarray, block_table_row: np.ndarray, seq_len: int, head: int) -> np.ndarray:
    # cache shape: [num_blocks, num_heads, block_size, head_dim]
    _, _, block_size, head_dim = cache.shape
    out = np.empty((seq_len, head_dim), dtype=cache.dtype)
    for pos in range(seq_len):
        logical = pos // block_size
        off = pos % block_size
        physical = int(block_table_row[logical])
        out[pos] = cache[physical, head, off]
    return out


def paged_decode_attention_reference(q: np.ndarray, k_cache: np.ndarray, v_cache: np.ndarray, block_table_row: np.ndarray, seq_len: int, head: int) -> np.ndarray:
    k = gather_paged_sequence(k_cache, block_table_row, seq_len, head)
    v = gather_paged_sequence(v_cache, block_table_row, seq_len, head)
    scores = k.astype(np.float32) @ q.astype(np.float32) / math.sqrt(q.size)
    p = np.exp(scores - np.max(scores)); p = p / np.sum(p)
    return p @ v.astype(np.float32)


def smoke_test() -> None:
    rng = np.random.default_rng(3)
    k_cache = rng.normal(size=(5, 2, 4, 8)).astype(np.float32)
    v_cache = rng.normal(size=(5, 2, 4, 8)).astype(np.float32)
    table = np.array([3, 1, 4], dtype=np.int32)
    q = rng.normal(size=(8,)).astype(np.float32)
    out = paged_decode_attention_reference(q, k_cache, v_cache, table, seq_len=10, head=1)
    assert out.shape == (8,)
