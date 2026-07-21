"""
CUDA execution model simulation for Transformer kernels.

Beginner:
    CUDA programs are organized as grids of CTAs/blocks. Each CTA contains threads.
    Threads are executed in groups of 32 called warps.

Intermediate:
    Attention kernels generally map one CTA to a tile of the attention matrix or to one
    query block. A warp usually owns a smaller tile inside that CTA.

Advanced:
    Coalescing matters. If lane 0 reads address A, lane 1 reads A+1, ..., lane 31 reads A+31,
    then the warp performs a coalesced memory transaction. If lanes read scattered addresses,
    bandwidth drops.

Expert:
    Tensor-core kernels organize work in CTA -> warp -> MMA instruction hierarchy.
    Each MMA instruction consumes register fragments that were loaded from global memory through
    shared memory, often using vectorized global loads and `ldmatrix` shared-memory loads.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

WARP_SIZE = 32

@dataclass(frozen=True)
class ThreadCoord:
    block_id: int
    thread_id: int
    warp_id_in_block: int
    lane_id: int


def decompose_thread(block_id: int, thread_id: int) -> ThreadCoord:
    """Map a CTA-local thread id to warp id and lane id."""
    return ThreadCoord(
        block_id=block_id,
        thread_id=thread_id,
        warp_id_in_block=thread_id // WARP_SIZE,
        lane_id=thread_id % WARP_SIZE,
    )


def linear_address(row: int, col: int, stride: int, element_bytes: int = 2) -> int:
    """Byte address for a row-major tensor element."""
    return (row * stride + col) * element_bytes


def warp_contiguous_load_addresses(base_row: int, base_col: int, stride: int, element_bytes: int = 2) -> list[int]:
    """Addresses loaded by one warp when every lane loads adjacent columns."""
    return [linear_address(base_row, base_col + lane, stride, element_bytes) for lane in range(WARP_SIZE)]


def count_memory_segments(addresses: list[int], segment_bytes: int = 128) -> int:
    """
    Count unique memory segments touched by a warp.

    Interview point:
        Fewer segments usually means better coalescing. For FP16, 32 lanes loading contiguous
        2-byte values touch 64 bytes. Depending alignment, that can fit in one 128-byte segment.
    """
    return len({addr // segment_bytes for addr in addresses})


def cta_tile_ownership(m: int, n: int, block_m: int, block_n: int) -> list[tuple[int, int, int, int]]:
    """Return tiles as (row_start, row_end, col_start, col_end)."""
    tiles = []
    for r in range(0, m, block_m):
        for c in range(0, n, block_n):
            tiles.append((r, min(r + block_m, m), c, min(c + block_n, n)))
    return tiles


def smoke_test() -> None:
    tc = decompose_thread(3, 70)
    assert tc.warp_id_in_block == 2 and tc.lane_id == 6
    contiguous = warp_contiguous_load_addresses(0, 0, stride=1024)
    strided = [linear_address(lane, 0, 1024) for lane in range(WARP_SIZE)]
    assert count_memory_segments(contiguous) <= count_memory_segments(strided)
    assert len(cta_tile_ownership(129, 65, 64, 32)) == math.ceil(129/64) * math.ceil(65/32)
