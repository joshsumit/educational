from __future__ import annotations
"""Stage 9.1 — Online softmax reference.

This file proves that blockwise online softmax can match normal stable softmax.

It computes softmax over a 1D vector by streaming blocks and maintaining:

    m: running max
    l: running denominator

For output probabilities, we need a second pass or we store/rescale numerators. FlashAttention avoids storing
probabilities by maintaining the running output accumulator directly.
"""

import math
import numpy as np


def stable_softmax(x: np.ndarray) -> np.ndarray:
    m = np.max(x)
    e = np.exp(x - m)
    return e / np.sum(e)


def online_softmax_stats(x: np.ndarray, block: int = 16) -> tuple[float, float]:
    m = -math.inf
    l = 0.0
    for start in range(0, x.size, block):
        xb = x[start:start+block].astype(np.float32)
        mb = float(np.max(xb))
        m_new = max(m, mb)
        old_scale = 0.0 if m == -math.inf else math.exp(m - m_new)
        p = np.exp(xb - m_new)
        l = l * old_scale + float(np.sum(p))
        m = m_new
    return m, l


def online_softmax_two_pass(x: np.ndarray, block: int = 16) -> np.ndarray:
    m, l = online_softmax_stats(x, block)
    out = np.empty_like(x, dtype=np.float32)
    for start in range(0, x.size, block):
        xb = x[start:start+block].astype(np.float32)
        out[start:start+block] = np.exp(xb - m) / l
    return out


def smoke_test() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(size=97).astype(np.float32) * 4
    assert np.allclose(online_softmax_two_pass(x, block=13), stable_softmax(x), atol=1e-6)
    m, l = online_softmax_stats(x, block=13)
    assert np.isfinite(m)
    assert l > 0
