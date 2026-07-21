from __future__ import annotations
"""Stage 13.7 — Profile report template.

Use this template after benchmarking a kernel.
"""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class KernelProfileReport:
    kernel: str
    input_shape: str
    dtype: str
    hardware: str
    baseline: str
    optimized: str
    warmup_iterations: int
    measurement_iterations: int
    latency_p50_ms: float
    latency_p95_ms: float
    throughput: str
    flops: float
    bytes_moved: float
    arithmetic_intensity: float
    likely_bottleneck: str
    next_tuning_action: str


def empty_report(kernel: str) -> KernelProfileReport:
    return KernelProfileReport(kernel, 'TODO', 'TODO', 'TODO', 'TODO', 'TODO', 25, 100, 0.0, 0.0, 'TODO', 0.0, 0.0, 0.0, 'TODO', 'TODO')


def report_to_markdown(report: KernelProfileReport) -> str:
    d = asdict(report)
    return '\n'.join(f'- {k}: {v}' for k,v in d.items())


def smoke_test() -> None:
    r = empty_report('matmul')
    md = report_to_markdown(r)
    assert '- kernel: matmul' in md
