"""Stage 1.3 — Grid mapping.

A Triton launch specifies a grid. For a 1D vector kernel:

    grid = (triton.cdiv(n, BLOCK),)

Meaning:

    launch enough programs to cover n elements, with BLOCK logical elements per program.

Example:

    n = 1000
    BLOCK = 256
    grid = (4,)

The comma matters in Python: `(4,)` is a one-element tuple.
"""
from __future__ import annotations

import math


def cdiv(a: int, b: int) -> int:
    """Equivalent to triton.cdiv for positive divisors."""
    if b <= 0:
        raise ValueError('divisor must be positive')
    return math.ceil(a / b)


def grid_1d(n: int, block: int) -> tuple[int]:
    """Return the 1D launch grid for n elements."""
    return (cdiv(n, block),)


def explain_grid_1d(n: int, block: int) -> str:
    """Return a readable launch-grid explanation."""
    programs = cdiv(n, block)
    lines = [f'grid=({programs},), n={n}, BLOCK={block}']
    for pid in range(programs):
        start = pid * block
        stop = min(start + block, n)
        lines.append(f'pid={pid} owns valid elements [{start}, {stop})')
    return '\n'.join(lines)


def smoke_test() -> None:
    assert cdiv(1000, 256) == 4
    assert grid_1d(1000, 256) == (4,)
    assert 'pid=3 owns valid elements [768, 1000)' in explain_grid_1d(1000, 256)
