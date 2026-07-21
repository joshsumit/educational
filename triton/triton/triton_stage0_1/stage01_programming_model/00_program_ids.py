"""Stage 1.0 — Program IDs.

The first Triton concept to master is `tl.program_id(axis)`.

A kernel is launched with a grid. For a 1D grid, every program gets one id:

    pid = 0
    pid = 1
    pid = 2
    ...

The program id is not the data index by itself. It is used to derive a block/tile of data indices.

For vector add:

    offsets = pid * BLOCK + arange(0, BLOCK)

For matmul later:

    pid_m = ...
    pid_n = ...

Interview answer:
    `program_id` is the coordinate of a Triton program instance in the launch grid. It is used to decide
    which block or tile of the input/output tensors this program owns.
"""
from __future__ import annotations


def program_ids_1d(num_programs: int) -> list[int]:
    """Return all program ids for a 1D launch."""
    if num_programs < 0:
        raise ValueError('num_programs must be non-negative')
    return list(range(num_programs))


def owner_pid_for_element(index: int, block: int) -> int:
    """Return which 1D program owns a given element index."""
    if index < 0:
        raise ValueError('index must be non-negative')
    if block <= 0:
        raise ValueError('block must be positive')
    return index // block


def program_start_offset(pid: int, block: int) -> int:
    """Return the first logical element offset owned by a 1D program."""
    return pid * block


def smoke_test() -> None:
    assert program_ids_1d(4) == [0, 1, 2, 3]
    assert owner_pid_for_element(0, 256) == 0
    assert owner_pid_for_element(255, 256) == 0
    assert owner_pid_for_element(256, 256) == 1
    assert program_start_offset(3, 256) == 768
