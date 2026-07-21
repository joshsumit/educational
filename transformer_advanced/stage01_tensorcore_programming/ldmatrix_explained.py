"""
`ldmatrix` explained through shared-memory tile swizzling.

Real role:
    `ldmatrix` loads matrix fragments from shared memory into registers in a layout expected by MMA.

Why it exists:
    Tensor cores expect operands in a lane-distributed register layout. Shared memory is the staging
    area between global memory and those registers.

This file simulates:
    - shared memory tile with optional padding
    - bank index calculation
    - a simplified lane-to-element load pattern
"""
from __future__ import annotations
import numpy as np

NUM_BANKS = 32
BANK_WIDTH_BYTES = 4


def shared_memory_bank(byte_address: int) -> int:
    """Bank id for a byte address under a simple 32-bank model."""
    return (byte_address // BANK_WIDTH_BYTES) % NUM_BANKS


def make_shared_tile(rows: int, cols: int, pad: int = 0) -> np.ndarray:
    """Create a row-major shared-memory tile with optional column padding."""
    return np.arange(rows * (cols + pad), dtype=np.float16).reshape(rows, cols + pad)


def lane_load_addresses(rows: int = 8, cols: int = 8, stride: int = 8, element_bytes: int = 2) -> dict[int, list[int]]:
    """
    Simplified lane load map for an 8x8 half tile.

    Not an exact SASS contract. This is a study model showing why lanes cooperatively load
    matrix elements from shared memory.
    """
    mapping = {lane: [] for lane in range(32)}
    element = 0
    for r in range(rows):
        for c in range(cols):
            lane = element % 32
            mapping[lane].append((r * stride + c) * element_bytes)
            element += 1
    return mapping


def bank_conflict_count(mapping: dict[int, list[int]]) -> int:
    """Count repeated bank hits among addresses loaded in the same simplified step."""
    banks = []
    for addrs in mapping.values():
        if addrs:
            banks.append(shared_memory_bank(addrs[0]))
    return len(banks) - len(set(banks))


def smoke_test() -> None:
    no_pad = lane_load_addresses(stride=8)
    padded = lane_load_addresses(stride=9)
    assert bank_conflict_count(no_pad) >= 0
    assert make_shared_tile(8,8,1).shape == (8,9)
    assert isinstance(bank_conflict_count(padded), int)
