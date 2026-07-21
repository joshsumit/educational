"""Stage 1.4 — 2D grids.

2D grids prepare you for matmul, attention score tiles, and image-like tensors.

A matrix C[M, N] can be divided into output tiles:

    BLOCK_M rows by BLOCK_N columns

Grid:

    grid_m = ceil(M / BLOCK_M)
    grid_n = ceil(N / BLOCK_N)
    grid = (grid_m, grid_n)

In a real Triton kernel you may use:

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

Alternatively, production matmul kernels often use a flattened 1D grid and map one pid into pid_m/pid_n
to control program ordering for cache reuse.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


def cdiv(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError('divisor must be positive')
    return math.ceil(a / b)


@dataclass(frozen=True)
class Tile2D:
    pid_m: int
    pid_n: int
    row_start: int
    row_stop: int
    col_start: int
    col_stop: int

    def shape(self) -> tuple[int, int]:
        return (self.row_stop - self.row_start, self.col_stop - self.col_start)


def grid_2d(m: int, n: int, block_m: int, block_n: int) -> tuple[int, int]:
    """Return the 2D tile grid shape."""
    return (cdiv(m, block_m), cdiv(n, block_n))


def tile_for_program(pid_m: int, pid_n: int, m: int, n: int, block_m: int, block_n: int) -> Tile2D:
    """Return the valid matrix region owned by one 2D program."""
    row_start = pid_m * block_m
    col_start = pid_n * block_n
    return Tile2D(
        pid_m=pid_m,
        pid_n=pid_n,
        row_start=row_start,
        row_stop=min(row_start + block_m, m),
        col_start=col_start,
        col_stop=min(col_start + block_n, n),
    )


def enumerate_tiles(m: int, n: int, block_m: int, block_n: int) -> list[Tile2D]:
    """Enumerate tiles in simple row-major order."""
    gm, gn = grid_2d(m, n, block_m, block_n)
    return [tile_for_program(pm, pn, m, n, block_m, block_n) for pm in range(gm) for pn in range(gn)]


def smoke_test() -> None:
    assert grid_2d(9, 7, 4, 3) == (3, 3)
    t = tile_for_program(2, 2, 9, 7, 4, 3)
    assert t.row_start == 8 and t.row_stop == 9
    assert t.col_start == 6 and t.col_stop == 7
    assert t.shape() == (1, 1)
    assert len(enumerate_tiles(9, 7, 4, 3)) == 9
