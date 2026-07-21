from __future__ import annotations
"""Stage 6.0 — Matmul shapes and strides.

Matmul convention:

    A[M, K] @ B[K, N] = C[M, N]

In a real Triton matmul kernel, the kernel usually receives raw pointers and strides:

    A address = a_ptr + row_m * stride_am + col_k * stride_ak
    B address = b_ptr + row_k * stride_bk + col_n * stride_bn
    C address = c_ptr + row_m * stride_cm + col_n * stride_cn

For contiguous row-major arrays:

    A strides: stride_am = K, stride_ak = 1
    B strides: stride_bk = N, stride_bn = 1
    C strides: stride_cm = N, stride_cn = 1

Why this matters:
    - Production kernels should not silently assume every tensor is contiguous.
    - Transposed views and packed layouts change strides.
    - Attention uses Q/K/V layouts where stride reasoning is critical.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MatmulShape:
    m: int
    n: int
    k: int

    @property
    def a_shape(self) -> tuple[int, int]:
        return (self.m, self.k)

    @property
    def b_shape(self) -> tuple[int, int]:
        return (self.k, self.n)

    @property
    def c_shape(self) -> tuple[int, int]:
        return (self.m, self.n)


@dataclass(frozen=True)
class MatrixStrides:
    stride_row: int
    stride_col: int

    def offset(self, row: int, col: int) -> int:
        return row * self.stride_row + col * self.stride_col


def row_major_strides(rows: int, cols: int) -> MatrixStrides:
    return MatrixStrides(stride_row=cols, stride_col=1)


def transposed_view_strides(original_rows: int, original_cols: int) -> MatrixStrides:
    """Strides of a transposed view of a row-major [original_rows, original_cols] matrix.

    Original row-major strides are [original_cols, 1].
    The transposed view shape is [original_cols, original_rows] with strides [1, original_cols].
    """
    return MatrixStrides(stride_row=1, stride_col=original_cols)


def validate_matmul_shapes(a: np.ndarray, b: np.ndarray) -> MatmulShape:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError('matmul inputs must be rank-2')
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(f'shape mismatch: A is {a.shape}, B is {b.shape}')
    return MatmulShape(m=m, n=n, k=k)


def matmul_flops(m: int, n: int, k: int) -> int:
    """Approximate FLOPs for C=A@B: multiply+add = 2 operations."""
    return 2 * m * n * k


def smoke_test() -> None:
    a = np.zeros((3, 5), dtype=np.float32)
    b = np.zeros((5, 7), dtype=np.float32)
    shape = validate_matmul_shapes(a, b)
    assert shape.a_shape == (3, 5)
    assert shape.b_shape == (5, 7)
    assert shape.c_shape == (3, 7)
    assert row_major_strides(3, 5).offset(2, 4) == 14
    assert transposed_view_strides(3, 5).offset(4, 2) == 14
    assert matmul_flops(3, 7, 5) == 210
