from __future__ import annotations
"""Stage 13.8 — Profiling interview notes."""


def roofline_answer() -> str:
    return 'Roofline compares arithmetic intensity against peak compute and memory bandwidth to estimate whether a kernel is compute-bound or memory-bound.'


def benchmark_answer() -> str:
    return 'A good benchmark warms up, synchronizes, excludes compile time, runs many iterations, reports shape/dtype/hardware, and compares against a baseline.'


def decode_profile_answer() -> str:
    return 'Decode attention should be analyzed in GB/s and KV bytes per generated token because it often streams the KV cache and is bandwidth-bound.'


def smoke_test() -> None:
    assert 'arithmetic intensity' in roofline_answer()
    assert 'synchronizes' in benchmark_answer()
    assert 'GB/s' in decode_profile_answer()
