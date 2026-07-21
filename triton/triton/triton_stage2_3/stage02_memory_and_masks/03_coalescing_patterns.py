"""Stage 2.3 — Coalescing patterns.

Memory coalescing means adjacent lanes load adjacent memory addresses. It is not the only performance factor,
but it is one of the first things to inspect.

Good pattern:
    offsets = [0, 1, 2, 3, 4, 5, 6, 7]

Strided pattern:
    offsets = [0, 4, 8, 12, 16, 20, 24, 28]

Bad/scattered pattern:
    offsets = [7, 2, 19, 3, 100, 5, 11, 8]

This file gives a small diagnostic score. It is not a hardware profiler. It is a study aid.
"""
from __future__ import annotations
import numpy as np


def adjacent_contiguity_score(offsets: np.ndarray) -> float:
    """Fraction of adjacent addresses that differ by exactly 1."""
    flat = offsets.reshape(-1)
    if flat.size < 2:
        return 1.0
    return float(np.mean(np.diff(flat) == 1))


def average_stride(offsets: np.ndarray) -> float:
    flat = offsets.reshape(-1)
    if flat.size < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(flat))))


def classify_access(offsets: np.ndarray) -> str:
    score = adjacent_contiguity_score(offsets)
    if score > 0.95:
        return 'contiguous/coalescing-friendly'
    if average_stride(offsets) <= 4:
        return 'small-stride'
    return 'scattered-or-large-stride'


def smoke_test() -> None:
    assert classify_access(np.arange(8)) == 'contiguous/coalescing-friendly'
    assert classify_access(np.arange(0, 32, 4)) == 'small-stride'
    assert classify_access(np.array([7, 2, 19, 3, 100])) == 'scattered-or-large-stride'
