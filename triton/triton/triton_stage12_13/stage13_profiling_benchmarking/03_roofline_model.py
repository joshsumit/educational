from __future__ import annotations
"""Stage 13.3 — Roofline model.

Roofline reasoning:

    arithmetic_intensity = FLOPs / bytes_moved
    compute_bound_limit = peak_flops
    bandwidth_bound_limit = arithmetic_intensity * memory_bandwidth
    achievable = min(compute_bound_limit, bandwidth_bound_limit)

If measured performance is much lower than roofline, investigate occupancy, memory coalescing, register pressure,
launch overhead, or poor tiling.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RooflineResult:
    arithmetic_intensity: float
    compute_limit_tflops: float
    bandwidth_limit_tflops: float
    predicted_tflops: float
    likely_bound: str


def roofline(flops: float, bytes_moved: float, peak_tflops: float, bandwidth_gbs: float) -> RooflineResult:
    ai = flops / max(bytes_moved, 1.0)
    bandwidth_limit = ai * bandwidth_gbs / 1000.0
    pred = min(peak_tflops, bandwidth_limit)
    bound = 'compute' if peak_tflops <= bandwidth_limit else 'memory'
    return RooflineResult(ai, peak_tflops, bandwidth_limit, pred, bound)


def smoke_test() -> None:
    r = roofline(1e12, 1e10, 100.0, 1000.0)
    assert r.arithmetic_intensity == 100.0
    assert r.predicted_tflops > 0
