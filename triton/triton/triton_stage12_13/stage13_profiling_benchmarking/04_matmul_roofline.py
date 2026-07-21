from __future__ import annotations
"""Stage 13.4 — Matmul roofline helpers."""

import importlib
_rf = importlib.import_module('stage13_profiling_benchmarking.03_roofline_model')
roofline = _rf.roofline
RooflineResult = _rf.RooflineResult


def matmul_flops(m: int, n: int, k: int) -> int:
    return 2 * m * n * k


def matmul_bytes(m: int, n: int, k: int, bytes_per_element: int = 2) -> int:
    return (m * k + k * n + m * n) * bytes_per_element


def matmul_roofline(m: int, n: int, k: int, peak_tflops: float, bandwidth_gbs: float, bytes_per_element: int = 2) -> RooflineResult:
    return roofline(matmul_flops(m,n,k), matmul_bytes(m,n,k,bytes_per_element), peak_tflops, bandwidth_gbs)


def smoke_test() -> None:
    r = matmul_roofline(1024,1024,1024, peak_tflops=100, bandwidth_gbs=1000)
    assert r.arithmetic_intensity > 100
