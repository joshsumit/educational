from __future__ import annotations
"""Stage 12.4 — INT4 packing reference.

Two signed int4 values fit into one uint8 byte.

Range:
    signed int4 = [-8, 7]

Packing convention in this file:
    low nibble  = first value
    high nibble = second value

For signed values, convert to 4-bit two's complement representation:
    -1 -> 0b1111 = 15
    -8 -> 0b1000 = 8
"""

import numpy as np


def int4_to_nibble(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -8, 7).astype(np.int8)
    return (x & 0x0F).astype(np.uint8)


def nibble_to_int4(x: np.ndarray) -> np.ndarray:
    x = (x & 0x0F).astype(np.int8)
    return np.where(x >= 8, x - 16, x).astype(np.int8)


def pack_int4(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(-1).astype(np.int8)
    if flat.size % 2 == 1:
        flat = np.concatenate([flat, np.zeros((1,), dtype=np.int8)])
    nib = int4_to_nibble(flat)
    low = nib[0::2]
    high = nib[1::2] << 4
    return (low | high).astype(np.uint8)


def unpack_int4(packed: np.ndarray, original_size: int | None = None) -> np.ndarray:
    packed = packed.astype(np.uint8).reshape(-1)
    low = nibble_to_int4(packed & 0x0F)
    high = nibble_to_int4((packed >> 4) & 0x0F)
    out = np.empty((packed.size * 2,), dtype=np.int8)
    out[0::2] = low; out[1::2] = high
    return out if original_size is None else out[:original_size]


def smoke_test() -> None:
    x = np.array([-8, -1, 0, 1, 7], dtype=np.int8)
    p = pack_int4(x)
    u = unpack_int4(p, original_size=x.size)
    assert u.tolist() == x.tolist()
