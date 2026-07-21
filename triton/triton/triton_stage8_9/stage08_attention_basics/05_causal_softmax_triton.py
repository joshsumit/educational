from __future__ import annotations
"""Stage 8.5 — Triton causal softmax for attention scores.

This is a row-wise softmax specialized to causal attention.

Program mapping:
    one program = one query row

Masking:
    cols <= row

This is still a materialized-score approach:
    scores [T,T] has already been written to memory.

FlashAttention later avoids writing and rereading this full score matrix.
"""

import numpy as np

try:
    import torch
    import triton
    import triton.language as tl
except Exception:
    torch = triton = tl = None


def causal_softmax_reference(scores: np.ndarray) -> np.ndarray:
    t = scores.shape[1]
    rows = np.arange(scores.shape[0])[:, None]
    cols = np.arange(t)[None, :]
    masked = np.where(cols <= rows, scores.astype(np.float32), -np.inf)
    m = np.max(masked, axis=1, keepdims=True)
    e = np.exp(masked - m)
    e = np.where(np.isfinite(masked), e, 0.0)
    return e / np.sum(e, axis=1, keepdims=True)


if tl is not None:
    @triton.jit
    def causal_softmax_kernel(scores_ptr, probs_ptr, T: tl.constexpr, stride_s: tl.constexpr, stride_p: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        in_bounds = cols < T
        causal = cols <= row
        mask = in_bounds & causal
        s = tl.load(scores_ptr + row * stride_s + cols, mask=in_bounds, other=-float('inf')).to(tl.float32)
        s = tl.where(mask, s, -float('inf'))
        s = s - tl.max(s, axis=0)
        p = tl.exp(s)
        p = tl.where(mask, p, 0.0)
        p = p / tl.sum(p, axis=0)
        tl.store(probs_ptr + row * stride_p + cols, p, mask=in_bounds)


def causal_softmax_triton(scores, block: int | None = None):
    if triton is None or torch is None:
        raise RuntimeError('Triton/Torch not available')
    TQ, TK = scores.shape
    if TQ != TK:
        raise ValueError('teaching causal kernel expects square [T,T] scores')
    block = block or (1 << (TK - 1).bit_length())
    probs = torch.empty_like(scores, dtype=torch.float32)
    causal_softmax_kernel[(TQ,)](scores, probs, TK, scores.stride(0), probs.stride(0), BLOCK=block)
    return probs


def smoke_test() -> None:
    scores = np.random.default_rng(3).normal(size=(8, 8)).astype(np.float32)
    p = causal_softmax_reference(scores)
    assert np.allclose(np.sum(p, axis=1), 1.0, atol=1e-6)
    assert np.allclose(p[np.triu_indices(8, k=1)], 0.0)

if __name__ == '__main__':
    smoke_test(); print('ok')
