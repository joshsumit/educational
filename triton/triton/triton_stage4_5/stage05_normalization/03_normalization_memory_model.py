from __future__ import annotations
"""Stage 5.3 — Normalization memory/performance model.

This file is not a profiler. It is a compact checklist to reason about LayerNorm/RMSNorm kernels in interviews.

Important questions:

1. How many global-memory streams are involved?
2. How many reductions per row?
3. Is accumulation fp32?
4. Is there a residual add that can be fused?
5. Is hidden size small enough for one-program-per-row?
6. Are gamma/beta/weight reads reused from cache?
"""


def layernorm_reduction_count() -> int:
    """LayerNorm usually needs mean and variance reductions."""
    return 2


def rmsnorm_reduction_count() -> int:
    """RMSNorm needs one sum-of-squares reduction."""
    return 1


def estimate_layernorm_bytes(m: int, n: int, bytes_per_element: int = 4) -> int:
    """Approximate logical traffic: read x, gamma, beta and write y.

    Gamma/beta are reused across rows in cache, but logical first-order accounting counts them per element.
    """
    return 4 * m * n * bytes_per_element


def estimate_rmsnorm_bytes(m: int, n: int, bytes_per_element: int = 4) -> int:
    """Approximate logical traffic: read x, read weight, write y."""
    return 3 * m * n * bytes_per_element


def choose_norm_kernel(hidden_size: int, has_residual: bool, norm_type: str) -> str:
    """Simple decision tree for study purposes."""
    if norm_type not in {'layernorm', 'rmsnorm'}:
        raise ValueError('norm_type must be layernorm or rmsnorm')
    if norm_type == 'rmsnorm' and has_residual:
        return 'fused_residual_rmsnorm'
    if norm_type == 'rmsnorm':
        return 'rmsnorm_one_program_per_row' if hidden_size <= 8192 else 'rmsnorm_split_row_or_persistent'
    return 'layernorm_one_program_per_row' if hidden_size <= 8192 else 'layernorm_split_row_or_persistent'


def smoke_test() -> None:
    assert layernorm_reduction_count() == 2
    assert rmsnorm_reduction_count() == 1
    assert estimate_layernorm_bytes(1, 10) == 160
    assert choose_norm_kernel(4096, True, 'rmsnorm') == 'fused_residual_rmsnorm'
