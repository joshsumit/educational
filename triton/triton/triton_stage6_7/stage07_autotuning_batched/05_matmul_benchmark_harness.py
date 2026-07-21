from __future__ import annotations
"""Stage 7.5 — Matmul benchmark harness.

This is a CPU-safe benchmark/report template. GPU benchmarking is intentionally not forced in smoke tests.

Good benchmark discipline:
    - warm up before measuring
    - synchronize GPU before and after timing
    - run multiple iterations
    - report p50/p95 if measuring latency distributions
    - compute TFLOP/s
    - state dtype and hardware
    - compare against torch.matmul or vendor BLAS where appropriate
"""

from dataclasses import dataclass, asdict
import time
import numpy as np


@dataclass(frozen=True)
class MatmulBenchmarkReport:
    m: int
    n: int
    k: int
    dtype: str
    implementation: str
    seconds: float
    flops: float
    tflops: float
    approximate_bytes: float
    arithmetic_intensity: float


def run_numpy_matmul_benchmark(m: int = 256, n: int = 256, k: int = 256, dtype: str = 'float32') -> MatmulBenchmarkReport:
    rng = np.random.default_rng(10)
    np_dtype = np.float32 if dtype == 'float32' else np.float16
    a = rng.normal(size=(m, k)).astype(np_dtype)
    b = rng.normal(size=(k, n)).astype(np_dtype)
    _ = a @ b  # warmup
    t0 = time.perf_counter()
    c = a @ b
    t1 = time.perf_counter()
    assert c.shape == (m, n)
    flops = float(2 * m * n * k)
    seconds = t1 - t0
    bytes_per_elem = 4 if dtype == 'float32' else 2
    approximate_bytes = float((m * k + k * n + m * n) * bytes_per_elem)
    return MatmulBenchmarkReport(
        m=m,
        n=n,
        k=k,
        dtype=dtype,
        implementation='numpy.matmul',
        seconds=seconds,
        flops=flops,
        tflops=flops / max(seconds, 1e-12) / 1e12,
        approximate_bytes=approximate_bytes,
        arithmetic_intensity=flops / approximate_bytes,
    )


def report_as_dict(report: MatmulBenchmarkReport) -> dict[str, float | int | str]:
    return asdict(report)


def smoke_test() -> None:
    report = run_numpy_matmul_benchmark(32, 32, 32)
    d = report_as_dict(report)
    assert d['m'] == 32
    assert d['flops'] == float(2 * 32 * 32 * 32)
    assert d['arithmetic_intensity'] > 0

if __name__ == '__main__':
    print(report_as_dict(run_numpy_matmul_benchmark()))
