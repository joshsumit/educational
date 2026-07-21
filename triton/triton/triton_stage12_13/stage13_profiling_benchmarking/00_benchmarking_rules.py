from __future__ import annotations
"""Stage 13.0 — Benchmarking rules.

Kernel benchmarking rules:

1. Warm up before measuring.
2. Synchronize GPU before starting and after ending timer.
3. Use enough iterations.
4. Report shape, dtype, hardware, and kernel config.
5. Compare against a meaningful baseline.
6. Report latency and throughput.
7. Compute FLOP/s or GB/s when relevant.
8. Do not benchmark debug builds or first-run compilation time.
"""


def benchmarking_checklist() -> list[str]:
    return ['warmup', 'synchronize', 'iterations', 'shape', 'dtype', 'hardware', 'baseline', 'latency', 'throughput']


def smoke_test() -> None:
    assert 'warmup' in benchmarking_checklist()
    assert 'baseline' in benchmarking_checklist()
