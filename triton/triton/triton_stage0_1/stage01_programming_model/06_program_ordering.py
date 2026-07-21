"""Stage 1.6 — Program ordering and grouped matmul order.

Program order can affect cache locality.

For simple vector add, order usually does not matter much. For matmul, order can affect reuse of matrix B
tiles in L2 cache.

Naive row-major tile order:

    (0,0), (0,1), (0,2),
    (1,0), (1,1), (1,2),
    (2,0), (2,1), (2,2)

Grouped order with GROUP_M=2:

    (0,0), (1,0),
    (0,1), (1,1),
    (0,2), (1,2),
    (2,0),
    (2,1),
    (2,2)

Why grouped order helps:
    Adjacent programs may reuse the same B tile while moving through nearby M tiles. This can improve L2
    cache reuse in tiled matmul kernels.
"""
from __future__ import annotations


def row_major_order(num_m: int, num_n: int) -> list[tuple[int, int]]:
    """Simple row-major tile order."""
    return [(m, n) for m in range(num_m) for n in range(num_n)]


def grouped_matmul_program_order(num_m: int, num_n: int, group_m: int) -> list[tuple[int, int]]:
    """Simulate Triton-style grouped ordering for matmul tiles.

    Args:
        num_m: number of tiles along M dimension
        num_n: number of tiles along N dimension
        group_m: how many M tiles to group for each sweep over N
    """
    if group_m <= 0:
        raise ValueError('group_m must be positive')
    order: list[tuple[int, int]] = []
    for group_start_m in range(0, num_m, group_m):
        actual_group_m = min(group_m, num_m - group_start_m)
        for n in range(num_n):
            for local_m in range(actual_group_m):
                order.append((group_start_m + local_m, n))
    return order


def flattened_to_grouped_tile(pid: int, num_m: int, num_n: int, group_m: int) -> tuple[int, int]:
    """Map a flattened pid to a grouped matmul tile coordinate.

    This mirrors the logic commonly found inside Triton matmul kernels.
    """
    num_pid_in_group = group_m * num_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_m
    group_size_m = min(num_m - first_pid_m, group_m)
    local = pid % num_pid_in_group
    pid_m = first_pid_m + (local % group_size_m)
    pid_n = local // group_size_m
    return pid_m, pid_n


def smoke_test() -> None:
    assert row_major_order(2, 3) == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    grouped = grouped_matmul_program_order(4, 2, 2)
    assert grouped == [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (3, 0), (2, 1), (3, 1)]
    mapped = [flattened_to_grouped_tile(pid, 4, 2, 2) for pid in range(8)]
    assert mapped == grouped
