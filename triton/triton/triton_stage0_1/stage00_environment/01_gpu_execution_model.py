"""Stage 0.1 — GPU execution model, simulated in Python.

This file is deliberately CPU-only. It teaches the launch-grid idea before any real Triton kernel appears.

Key idea:
    A GPU kernel launch creates many independent pieces of work.

Triton vocabulary:
    - program: an independent instance of a Triton kernel
    - program_id: integer coordinate identifying one program
    - block: the chunk/tile of data owned by one program

CUDA rough equivalent:
    - CUDA blockIdx.x is conceptually close to tl.program_id(0)
    - CUDA blockDim/threadIdx are hidden behind vectorized operations such as tl.arange

Example:
    N = 1000
    BLOCK = 256

    Number of programs = ceil(1000 / 256) = 4

    Program 0 handles offsets 0..255
    Program 1 handles offsets 256..511
    Program 2 handles offsets 512..767
    Program 3 handles offsets 768..999, with masks for 1000..1023
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ProgramRange:
    """The logical ownership range of one program.

    `start` is inclusive. `stop` is exclusive. This convention mirrors Python slices and makes it easy
    to reason about the elements owned by each program.
    """

    pid: int
    start: int
    stop: int
    block_size: int
    n_elements: int

    @property
    def valid_count(self) -> int:
        """Number of real elements this program owns.

        The last program often owns fewer than `block_size` valid elements.
        """
        return max(0, self.stop - self.start)

    @property
    def has_boundary_mask(self) -> bool:
        """True when this program's logical block extends beyond the tensor length."""
        return self.valid_count < self.block_size

    def offsets(self) -> list[int]:
        """All logical offsets a Triton program would create with tl.arange.

        These include invalid offsets for the last block. A real Triton kernel must use a mask before
        reading or writing those invalid locations.
        """
        base = self.pid * self.block_size
        return list(range(base, base + self.block_size))

    def mask(self) -> list[bool]:
        """Boundary mask equivalent to `offs < n_elements`."""
        return [offset < self.n_elements for offset in self.offsets()]


def ceil_div(a: int, b: int) -> int:
    """Integer ceil division, equivalent to triton.cdiv(a, b)."""
    if b <= 0:
        raise ValueError('divisor must be positive')
    return math.ceil(a / b)


def launch_1d(n_elements: int, block_size: int) -> list[ProgramRange]:
    """Return the per-program ownership ranges for a 1D launch."""
    if n_elements < 0:
        raise ValueError('n_elements must be non-negative')
    if block_size <= 0:
        raise ValueError('block_size must be positive')

    n_programs = ceil_div(n_elements, block_size)
    programs: list[ProgramRange] = []
    for pid in range(n_programs):
        start = pid * block_size
        stop = min(start + block_size, n_elements)
        programs.append(ProgramRange(pid, start, stop, block_size, n_elements))
    return programs


def visualize_launch(n_elements: int, block_size: int) -> str:
    """Human-readable launch map for interview explanation."""
    lines = [f'N={n_elements}, BLOCK={block_size}, programs={ceil_div(n_elements, block_size)}']
    for p in launch_1d(n_elements, block_size):
        boundary = ' boundary-mask-needed' if p.has_boundary_mask else ''
        lines.append(f'pid={p.pid}: valid [{p.start}, {p.stop}){boundary}')
    return '\n'.join(lines)


def smoke_test() -> None:
    programs = launch_1d(1000, 256)
    assert len(programs) == 4
    assert programs[0].start == 0 and programs[0].stop == 256
    assert programs[3].start == 768 and programs[3].stop == 1000
    assert programs[3].has_boundary_mask is True
    assert programs[3].mask().count(True) == 232
    assert ceil_div(1000, 256) == 4
