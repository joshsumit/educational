from __future__ import annotations
"""Stage 4.3 — Masked / causal row-softmax.

Goal:
    Softmax with a boolean mask.

This is the bridge from ordinary softmax to attention:

    scores = QK^T / sqrt(D)
    scores = apply causal/padding mask
    probs = softmax(scores)

Masking rule:
    masked scores become -inf before the max reduction.

Why not set masked scores to 0?
    Because exp(0) = 1, so masked positions would receive probability mass.

This file provides a CPU reference and a real Triton row-wise masked softmax kernel.
"""

import numpy as np

try:
    import torch
    import triton
    import triton.language as tl
except Exception:
    torch = triton = tl = None


def causal_mask(n: int) -> np.ndarray:
    """Lower triangular attention mask."""
    return np.tril(np.ones((n, n), dtype=bool))


def masked_softmax_reference(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    s = scores.astype(np.float32)
    masked = np.where(mask, s, -np.inf)
    row_max = np.max(masked, axis=1, keepdims=True)
    e = np.exp(masked - row_max)
    e = np.where(mask, e, 0.0)
    return e / np.sum(e, axis=1, keepdims=True)


def causal_softmax_reference(scores: np.ndarray) -> np.ndarray:
    return masked_softmax_reference(scores, causal_mask(scores.shape[1]))


if tl is not None:
    @triton.jit
    def causal_softmax_kernel(scores_ptr, out_ptr, N: tl.constexpr, stride_s: tl.constexpr, stride_o: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        in_bounds = cols < N
        causal = cols <= row
        mask = in_bounds & causal
        s = tl.load(scores_ptr + row * stride_s + cols, mask=in_bounds, other=-float('inf')).to(tl.float32)
        s = tl.where(mask, s, -float('inf'))
        s = s - tl.max(s, axis=0)
        p = tl.exp(s)
        p = tl.where(mask, p, 0.0)
        p = p / tl.sum(p, axis=0)
        tl.store(out_ptr + row * stride_o + cols, p, mask=in_bounds)


def causal_softmax_triton(scores, block: int | None = None):
    if triton is None or torch is None:
        raise RuntimeError('Triton/Torch not available')
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError('scores must be square [T, T] for this teaching causal kernel')
    N = scores.shape[1]
    block = block or (1 << (N - 1).bit_length())
    out = torch.empty_like(scores, dtype=torch.float32)
    causal_softmax_kernel[(N,)](scores, out, N, scores.stride(0), out.stride(0), BLOCK=block)
    return out


def smoke_test() -> None:
    scores = np.random.default_rng(5).normal(size=(11, 11)).astype(np.float32)
    probs = causal_softmax_reference(scores)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-6)
    assert np.allclose(probs[np.triu_indices(11, k=1)], 0.0)

if __name__ == '__main__':
    smoke_test(); print('ok')
