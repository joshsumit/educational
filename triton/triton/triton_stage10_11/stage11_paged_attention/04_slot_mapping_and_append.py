from __future__ import annotations
"""Stage 11.4 — Slot mapping and appending tokens to paged KV cache.

Serving runtimes often flatten active token writes into a slot mapping:

    slot_mapping[row] = physical_slot

For paged KV:

    physical_slot = physical_block * block_size + offset_in_block

This tells the KV write kernel where each new token's K/V vectors should be stored.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Slot:
    physical_block: int
    offset_in_block: int
    physical_slot: int


def compute_slot(block_table_row: list[int], position: int, block_size: int) -> Slot:
    logical = position // block_size
    off = position % block_size
    physical = block_table_row[logical]
    return Slot(physical, off, physical * block_size + off)


def build_slot_mapping(block_tables: list[list[int]], request_ids: list[int], positions: list[int], block_size: int) -> list[int]:
    if len(request_ids) != len(positions):
        raise ValueError('request_ids and positions must have same length')
    slots = []
    for rid, pos in zip(request_ids, positions):
        slots.append(compute_slot(block_tables[rid], pos, block_size).physical_slot)
    return slots


def smoke_test() -> None:
    tables = [[10, 11], [20, 21]]
    s = compute_slot(tables[1], 5, 4)
    assert s.physical_block == 21 and s.offset_in_block == 1 and s.physical_slot == 85
    assert build_slot_mapping(tables, [0, 1], [3, 5], 4) == [43, 85]
