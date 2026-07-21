"""
WMMA-style fragment simulation.

WMMA programming exposes matrix fragments:
    - matrix_a fragment
    - matrix_b fragment
    - accumulator fragment

Each fragment is conceptually a tile but physically distributed across warp lanes.
This file models fragments as normal NumPy arrays plus metadata.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class Fragment:
    role: str
    shape: tuple[int, int]
    dtype: str
    values: np.ndarray


def load_matrix_a(tile: np.ndarray) -> Fragment:
    if tile.shape != (16,16):
        raise ValueError('matrix_a fragment uses [16,16] in this reference')
    return Fragment('matrix_a', tile.shape, str(tile.dtype), tile.copy())


def load_matrix_b(tile: np.ndarray) -> Fragment:
    if tile.shape != (16,8):
        raise ValueError('matrix_b fragment uses [16,8] in this reference')
    return Fragment('matrix_b', tile.shape, str(tile.dtype), tile.copy())


def fill_accumulator(value: float = 0.0) -> Fragment:
    return Fragment('accumulator', (16,8), 'float32', np.full((16,8), value, dtype=np.float32))


def mma(a: Fragment, b: Fragment, acc: Fragment) -> Fragment:
    if a.role != 'matrix_a' or b.role != 'matrix_b' or acc.role != 'accumulator':
        raise ValueError('invalid fragment roles')
    out = a.values.astype(np.float32) @ b.values.astype(np.float32) + acc.values
    return Fragment('accumulator', out.shape, 'float32', out)


def store_matrix(acc: Fragment) -> np.ndarray:
    if acc.role != 'accumulator':
        raise ValueError('only accumulator fragments can be stored as output')
    return acc.values.copy()


def smoke_test() -> None:
    a_np = np.ones((16,16), dtype=np.float16)
    b_np = np.ones((16,8), dtype=np.float16)
    out = store_matrix(mma(load_matrix_a(a_np), load_matrix_b(b_np), fill_accumulator()))
    assert out.shape == (16,8)
    assert np.allclose(out, 16.0)
