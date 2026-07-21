from __future__ import annotations
"""Stage 8.2 — Causal masking reference.

Causal attention rule:

    query position i can attend to key positions j <= i

For square prefill attention [T,T], this is the lower triangular mask.

Masking rule before softmax:

    scores = where(mask, scores, -inf)

Never use 0 for masked logits before softmax. exp(0)=1 and would leak probability mass into masked positions.
"""

import numpy as np


def causal_mask(tq: int, tk: int | None = None) -> np.ndarray:
    tk = tq if tk is None else tk
    rows = np.arange(tq)[:, None]
    cols = np.arange(tk)[None, :]
    return cols <= rows


def apply_causal_mask(scores: np.ndarray) -> np.ndarray:
    tq, tk = scores.shape
    return np.where(causal_mask(tq, tk), scores.astype(np.float32), -np.inf)


def stable_softmax(x: np.ndarray) -> np.ndarray:
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    e = np.where(np.isfinite(x), e, 0.0)
    return e / np.sum(e, axis=-1, keepdims=True)


def causal_softmax(scores: np.ndarray) -> np.ndarray:
    return stable_softmax(apply_causal_mask(scores))


def smoke_test() -> None:
    s = np.arange(16, dtype=np.float32).reshape(4, 4)
    p = causal_softmax(s)
    assert np.allclose(np.sum(p, axis=1), 1.0)
    assert np.allclose(p[np.triu_indices(4, k=1)], 0.0)
    assert causal_mask(2, 4).tolist() == [[True, False, False, False], [True, True, False, False]]
