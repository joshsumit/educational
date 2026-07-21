from __future__ import annotations
"""Stage 6.6 — Matmul interview notes as executable facts.

These are the answers you should be able to give before discussing attention kernels.
"""


def matmul_shape_answer() -> str:
    return 'A[M,K] times B[K,N] produces C[M,N]. K is the reduction dimension.'


def tiled_matmul_answer() -> str:
    return 'One program computes a C tile, loops over K tiles, loads A/B tiles, accumulates with tl.dot, then stores C.'


def grouped_order_answer() -> str:
    return 'Grouped ordering changes program traversal to improve L2 reuse, commonly reuse of B tiles across nearby M tiles.'


def matmul_arithmetic_intensity(m: int, n: int, k: int, bytes_per_element: int = 2) -> float:
    """Rough arithmetic intensity for C=A@B.

    Assumes reading A and B once and writing C once. Real kernels have cache effects and tile reuse.
    """
    flops = 2 * m * n * k
    bytes_moved = (m * k + k * n + m * n) * bytes_per_element
    return flops / bytes_moved


def smoke_test() -> None:
    assert 'C[M,N]' in matmul_shape_answer()
    assert 'tl.dot' in tiled_matmul_answer()
    assert matmul_arithmetic_intensity(128, 128, 128, 2) > 40
