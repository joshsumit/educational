"""Stage 2.2 — Strides and layouts.

Interviews often ask: "What is the memory layout?" This file builds the vocabulary before matmul/attention.

Row-major [M, N]:
    stride_m = N
    stride_n = 1

Column-major [M, N]:
    stride_m = 1
    stride_n = M

Batched row-major [B, M, N]:
    stride_b = M * N
    stride_m = N
    stride_n = 1

Triton kernels often avoid assuming contiguity by accepting explicit strides.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Strided2D:
    rows: int
    cols: int
    stride_row: int
    stride_col: int

    def offset(self, row: int, col: int) -> int:
        return row * self.stride_row + col * self.stride_col


def row_major_layout(rows: int, cols: int) -> Strided2D:
    return Strided2D(rows, cols, stride_row=cols, stride_col=1)


def column_major_layout(rows: int, cols: int) -> Strided2D:
    return Strided2D(rows, cols, stride_row=1, stride_col=rows)


def offset_grid(layout: Strided2D) -> np.ndarray:
    rr = np.arange(layout.rows)[:, None]
    cc = np.arange(layout.cols)[None, :]
    return rr * layout.stride_row + cc * layout.stride_col


def smoke_test() -> None:
    rm = row_major_layout(2, 3)
    cm = column_major_layout(2, 3)
    assert offset_grid(rm).tolist() == [[0, 1, 2], [3, 4, 5]]
    assert offset_grid(cm).tolist() == [[0, 2, 4], [1, 3, 5]]
