from __future__ import annotations
"""Stage 10.4 — Multi-head decode reference.

Shape convention:

    q        [H, D]
    k_cache  [H, S, D]
    v_cache  [H, S, DV]
    out      [H, DV]

Production runtimes often combine sequence and head dimensions in kernel grids:

    program axis 0 -> sequence/head row
    program axis 1 -> optional tile dimension
"""

import math
import numpy as np


def decode_one(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    scores = k.astype(np.float32) @ q.astype(np.float32) / math.sqrt(q.size)
    p = np.exp(scores - np.max(scores)); p = p / np.sum(p)
    return p @ v.astype(np.float32)


def multihead_decode_reference(q: np.ndarray, k_cache: np.ndarray, v_cache: np.ndarray) -> np.ndarray:
    if q.ndim != 2 or k_cache.ndim != 3 or v_cache.ndim != 3:
        raise ValueError('expected q[H,D], k[H,S,D], v[H,S,DV]')
    h, d = q.shape
    h2, s, d2 = k_cache.shape
    h3, s2, dv = v_cache.shape
    if h != h2 or h != h3 or d != d2 or s != s2:
        raise ValueError('shape mismatch')
    out = np.empty((h, dv), dtype=np.float32)
    for head in range(h):
        out[head] = decode_one(q[head], k_cache[head], v_cache[head])
    return out


def smoke_test() -> None:
    rng = np.random.default_rng(2)
    q = rng.normal(size=(4, 16)).astype(np.float32)
    k = rng.normal(size=(4, 29, 16)).astype(np.float32)
    v = rng.normal(size=(4, 29, 12)).astype(np.float32)
    assert multihead_decode_reference(q, k, v).shape == (4, 12)
