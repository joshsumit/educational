from __future__ import annotations
"""Stage 8.0 — Attention shapes.

Single-head attention convention:

    Q[Tq, Dh]
    K[Tk, Dh]
    V[Tk, Dv]

    scores = Q @ K.T / sqrt(Dh)      -> [Tq, Tk]
    probs = softmax(scores, axis=-1) -> [Tq, Tk]
    O = probs @ V                    -> [Tq, Dv]

Multi-head layout commonly appears as:

    [B, H, T, Dh]

For many kernels, batch and head can be flattened:

    [B, H, T, Dh] -> [B*H, T, Dh]

Why this matters:
    - QK is a matmul with B transposed logically, not necessarily physically copied.
    - Causal attention requires masking future keys for each query row.
    - Decode attention has Tq=1 and large Tk, making it memory-bandwidth heavy.
"""

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class AttentionShape:
    tq: int
    tk: int
    dh: int
    dv: int

    @property
    def q_shape(self) -> tuple[int, int]:
        return (self.tq, self.dh)

    @property
    def k_shape(self) -> tuple[int, int]:
        return (self.tk, self.dh)

    @property
    def v_shape(self) -> tuple[int, int]:
        return (self.tk, self.dv)

    @property
    def scores_shape(self) -> tuple[int, int]:
        return (self.tq, self.tk)

    @property
    def output_shape(self) -> tuple[int, int]:
        return (self.tq, self.dv)

    @property
    def scale(self) -> float:
        return 1.0 / math.sqrt(self.dh)


def validate_attention_shapes(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> AttentionShape:
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError('single-head attention expects rank-2 Q/K/V')
    tq, dh = q.shape
    tk, dh2 = k.shape
    tk2, dv = v.shape
    if dh != dh2 or tk != tk2:
        raise ValueError('shape mismatch among Q/K/V')
    return AttentionShape(tq=tq, tk=tk, dh=dh, dv=dv)


def flatten_batch_head(x: np.ndarray) -> np.ndarray:
    """Flatten [B,H,T,D] into [B*H,T,D]."""
    if x.ndim != 4:
        raise ValueError('expected [B,H,T,D]')
    b, h, t, d = x.shape
    return x.reshape(b * h, t, d)


def smoke_test() -> None:
    q = np.zeros((5, 7), dtype=np.float32)
    k = np.zeros((9, 7), dtype=np.float32)
    v = np.zeros((9, 3), dtype=np.float32)
    s = validate_attention_shapes(q, k, v)
    assert s.scores_shape == (5, 9)
    assert s.output_shape == (5, 3)
    x = np.zeros((2, 4, 8, 16), dtype=np.float32)
    assert flatten_batch_head(x).shape == (8, 8, 16)
