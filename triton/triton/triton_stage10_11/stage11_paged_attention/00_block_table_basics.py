from __future__ import annotations
"""Stage 11.0 — Block table basics.

Paged attention replaces one large contiguous KV allocation per sequence with fixed-size physical blocks.

Terms:

    block_size:
        number of tokens per KV block/page

    logical block id:
        token_position // block_size

    offset within block:
        token_position % block_size

    block table:
        maps [request_id, logical_block_id] -> physical_block_id

Why this matters:
    - avoids large contiguous allocations
    - supports dynamic batching
    - supports memory reuse/eviction
    - enables prefix sharing in serving systems
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LogicalTokenLocation:
    logical_block: int
    offset_in_block: int


def locate_token(position: int, block_size: int) -> LogicalTokenLocation:
    if position < 0 or block_size <= 0:
        raise ValueError('invalid position/block_size')
    return LogicalTokenLocation(position // block_size, position % block_size)


def block_table_lookup(block_table: list[list[int]], request_id: int, position: int, block_size: int) -> tuple[int, int]:
    loc = locate_token(position, block_size)
    return block_table[request_id][loc.logical_block], loc.offset_in_block


def smoke_test() -> None:
    loc = locate_token(37, 16)
    assert loc.logical_block == 2 and loc.offset_in_block == 5
    table = [[10, 11, 12], [20, 21, 22]]
    assert block_table_lookup(table, 1, 35, 16) == (22, 3)
