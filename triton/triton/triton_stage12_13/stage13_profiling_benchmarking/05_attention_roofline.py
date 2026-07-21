from __future__ import annotations
"""Stage 13.5 — Attention roofline helpers."""

import importlib
_rf = importlib.import_module('stage13_profiling_benchmarking.03_roofline_model')
roofline = _rf.roofline
RooflineResult = _rf.RooflineResult


def attention_flops(t: int, d: int, dv: int | None = None) -> int:
    dv = d if dv is None else dv
    return 2*t*t*d + 2*t*t*dv


def materialized_attention_bytes(t: int, d: int, dv: int | None = None, bytes_per_element: int = 2) -> int:
    dv = d if dv is None else dv
    qkv_out = (t*d + t*d + t*dv + t*dv) * bytes_per_element
    scores_probs_traffic = 4 * t * t * bytes_per_element
    return qkv_out + scores_probs_traffic


def attention_roofline(t: int, d: int, peak_tflops: float, bandwidth_gbs: float, bytes_per_element: int = 2) -> RooflineResult:
    return roofline(attention_flops(t,d), materialized_attention_bytes(t,d,bytes_per_element=bytes_per_element), peak_tflops, bandwidth_gbs)


def smoke_test() -> None:
    r = attention_roofline(512, 64, 100, 1000)
    assert r.predicted_tflops > 0
