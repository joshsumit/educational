from __future__ import annotations
"""Stage 11.1 — Paged KV cache layout.

A simple physical paged KV cache can be represented as:

    K[num_blocks, num_heads, block_size, head_dim]
    V[num_blocks, num_heads, block_size, head_dim]

To access token position p for request r:

    logical_block = p // block_size
    offset        = p % block_size
    physical      = block_table[r, logical_block]
    K value       = K[physical, head, offset, dim]

This file implements CPU address math for that layout.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PagedKVLayout:
    num_blocks: int
    num_heads: int
    block_size: int
    head_dim: int

    @property
    def stride_block(self) -> int:
        return self.num_heads * self.block_size * self.head_dim

    @property
    def stride_head(self) -> int:
        return self.block_size * self.head_dim

    @property
    def stride_token(self) -> int:
        return self.head_dim

    @property
    def stride_dim(self) -> int:
        return 1

    def offset(self, physical_block: int, head: int, offset_in_block: int, dim: int) -> int:
        return (physical_block * self.stride_block + head * self.stride_head + offset_in_block * self.stride_token + dim)


def gather_sequence_for_head(cache_flat: np.ndarray, layout: PagedKVLayout, block_table_row: list[int], seq_len: int, head: int) -> np.ndarray:
    out = np.empty((seq_len, layout.head_dim), dtype=cache_flat.dtype)
    for pos in range(seq_len):
        logical = pos // layout.block_size
        off = pos % layout.block_size
        physical = block_table_row[logical]
        for d in range(layout.head_dim):
            out[pos, d] = cache_flat[layout.offset(physical, head, off, d)]
    return out


def smoke_test() -> None:
    layout = PagedKVLayout(num_blocks=4, num_heads=2, block_size=3, head_dim=4)
    assert layout.offset(1, 1, 2, 3) == 47
    flat = np.arange(4 * 2 * 3 * 4, dtype=np.float32)
    out = gather_sequence_for_head(flat, layout, [2, 0], seq_len=5, head=1)
    assert out.shape == (5, 4)
    assert out[0, 0] == flat[layout.offset(2, 1, 0, 0)]
