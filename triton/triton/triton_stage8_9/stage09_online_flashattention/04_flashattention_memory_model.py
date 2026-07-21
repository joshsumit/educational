from __future__ import annotations
"""Stage 9.4 — FlashAttention memory model.

Materialized attention writes/reads large [T,T] matrices:

    scores [T,T]
    probabilities [T,T]

FlashAttention avoids those global-memory matrices by keeping softmax stats and output accumulator on-chip per
tile.

This file gives first-order memory estimates. It is not a profiler.
"""


def materialized_attention_bytes(t: int, d: int, dv: int | None = None, bytes_per_element: int = 2) -> dict[str, int]:
    dv = d if dv is None else dv
    qkv = (t * d + t * d + t * dv) * bytes_per_element
    scores_write_read = 2 * t * t * bytes_per_element
    probs_write_read = 2 * t * t * bytes_per_element
    out = t * dv * bytes_per_element
    return {'qkv_bytes': qkv, 'score_matrix_traffic': scores_write_read, 'prob_matrix_traffic': probs_write_read, 'out_bytes': out, 'total_bytes': qkv + scores_write_read + probs_write_read + out}


def flashattention_first_order_bytes(t: int, d: int, dv: int | None = None, bytes_per_element: int = 2) -> dict[str, int]:
    """Very rough traffic estimate without full [T,T] global-memory matrices."""
    dv = d if dv is None else dv
    q_read = t * d * bytes_per_element
    # K/V are streamed; real traffic depends on tiling/cache. Count one logical pass for first-order discussion.
    k_read = t * d * bytes_per_element
    v_read = t * dv * bytes_per_element
    out_write = t * dv * bytes_per_element
    return {'q_read': q_read, 'k_read_logical': k_read, 'v_read_logical': v_read, 'out_write': out_write, 'total_logical_bytes': q_read + k_read + v_read + out_write}


def memory_savings_ratio(t: int, d: int, bytes_per_element: int = 2) -> float:
    mat = materialized_attention_bytes(t, d, bytes_per_element=bytes_per_element)['total_bytes']
    fla = flashattention_first_order_bytes(t, d, bytes_per_element=bytes_per_element)['total_logical_bytes']
    return mat / fla


def smoke_test() -> None:
    mat = materialized_attention_bytes(1024, 64)
    fla = flashattention_first_order_bytes(1024, 64)
    assert mat['total_bytes'] > fla['total_logical_bytes']
    assert memory_savings_ratio(1024, 64) > 1.0
