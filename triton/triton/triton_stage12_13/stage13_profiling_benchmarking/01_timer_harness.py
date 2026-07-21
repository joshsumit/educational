from __future__ import annotations
"""Stage 13.1 — CPU-safe timer harness.

This timer is for references and smoke tests. GPU kernels need synchronization and preferably CUDA events or
Triton benchmarking helpers.
"""

from dataclasses import dataclass
import time, statistics
from typing import Callable, Any


@dataclass(frozen=True)
class TimingStats:
    iterations: int
    mean_seconds: float
    median_seconds: float
    min_seconds: float
    max_seconds: float


def time_function(fn: Callable[[], Any], warmup: int = 3, iterations: int = 10) -> TimingStats:
    for _ in range(warmup): fn()
    samples=[]
    for _ in range(iterations):
        t0=time.perf_counter(); fn(); t1=time.perf_counter(); samples.append(t1-t0)
    return TimingStats(iterations, statistics.mean(samples), statistics.median(samples), min(samples), max(samples))


def smoke_test() -> None:
    x = 0
    def f():
        nonlocal x; x += 1
    stats = time_function(f, warmup=1, iterations=3)
    assert stats.iterations == 3 and x == 4
