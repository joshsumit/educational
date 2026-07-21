"""Stage 1.5 — 3D grids.

3D grids prepare you for:

    - batched matmul
    - multi-head attention
    - tensors shaped [batch, head, sequence, dim]

Example mapping for attention:

    axis 0: query/output tile id
    axis 1: head id
    axis 2: batch id

In real Triton:

    pid_tile  = tl.program_id(0)
    pid_head  = tl.program_id(1)
    pid_batch = tl.program_id(2)

This avoids manually flattening batch/head/tile into one integer when independent grid axes are clearer.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Program3D:
    tile: int
    head: int
    batch: int


def grid_3d(num_tiles: int, num_heads: int, batch_size: int) -> tuple[int, int, int]:
    """Return a 3D grid shape."""
    if num_tiles < 0 or num_heads < 0 or batch_size < 0:
        raise ValueError('grid dimensions must be non-negative')
    return (num_tiles, num_heads, batch_size)


def enumerate_programs_3d(num_tiles: int, num_heads: int, batch_size: int) -> list[Program3D]:
    """Enumerate all 3D program coordinates."""
    return [
        Program3D(tile=t, head=h, batch=b)
        for b in range(batch_size)
        for h in range(num_heads)
        for t in range(num_tiles)
    ]


def flatten_3d(tile: int, head: int, batch: int, num_tiles: int, num_heads: int) -> int:
    """Flatten a 3D coordinate into a single program id.

    Useful when comparing 3D grids with flattened production kernels.
    """
    return batch * (num_heads * num_tiles) + head * num_tiles + tile


def unflatten_3d(pid: int, num_tiles: int, num_heads: int) -> Program3D:
    """Inverse of flatten_3d."""
    batch = pid // (num_heads * num_tiles)
    rem = pid % (num_heads * num_tiles)
    head = rem // num_tiles
    tile = rem % num_tiles
    return Program3D(tile=tile, head=head, batch=batch)


def smoke_test() -> None:
    assert grid_3d(5, 8, 2) == (5, 8, 2)
    programs = enumerate_programs_3d(2, 3, 2)
    assert len(programs) == 12
    pid = flatten_3d(tile=1, head=2, batch=1, num_tiles=2, num_heads=3)
    assert pid == 11
    assert unflatten_3d(pid, num_tiles=2, num_heads=3) == Program3D(tile=1, head=2, batch=1)
