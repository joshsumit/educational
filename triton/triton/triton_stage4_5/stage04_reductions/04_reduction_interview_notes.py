from __future__ import annotations
"""Stage 4.4 — Reduction interview notes as executable checks.

This file captures the practical rules that keep coming up in kernel interviews.
"""

import numpy as np


def neutral_element(op: str) -> float:
    """Return the correct invalid-lane fill value for common reductions."""
    if op == 'sum':
        return 0.0
    if op == 'max':
        return -float('inf')
    if op == 'min':
        return float('inf')
    raise ValueError(f'unknown op: {op}')


def softmax_mask_fill_value() -> float:
    """Masked softmax positions should use -inf before exponentiation."""
    return -float('inf')


def estimate_row_softmax_work(m: int, n: int, bytes_per_element: int = 4) -> dict[str, float]:
    """Simple educational model for row-softmax traffic and arithmetic.

    This is not a profiler. It is a first-order checklist.
    """
    reads = m * n * bytes_per_element
    writes = m * n * bytes_per_element
    # Very rough: max compare, subtract, exp, sum, divide per element.
    operations = 5 * m * n
    return {
        'bytes_read': float(reads),
        'bytes_written': float(writes),
        'total_bytes': float(reads + writes),
        'rough_scalar_ops': float(operations),
    }


def smoke_test() -> None:
    assert neutral_element('sum') == 0.0
    assert np.isneginf(neutral_element('max'))
    assert np.isposinf(neutral_element('min'))
    model = estimate_row_softmax_work(2, 4)
    assert model['total_bytes'] == 64.0
