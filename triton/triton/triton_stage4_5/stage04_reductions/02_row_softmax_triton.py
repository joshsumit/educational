from __future__ import annotations
"""Stage 4.2 — Row-wise softmax in Triton.

Goal:
    For X[M, N], compute softmax independently for each row.

Stable softmax formula:

    row_max = max(x)
    exps = exp(x - row_max)
    denom = sum(exps)
    out = exps / denom

Why subtract max?
    It prevents overflow for large positive logits.

Program mapping:
    One Triton program handles one row.

Constraints of this simple teaching kernel:
    - N must be <= BLOCK.
    - BLOCK is usually a power of two for efficient compilation.
    - Later FlashAttention kernels avoid materializing full softmax matrices by doing online softmax.

Interview notes:
    - Invalid lanes load -inf before max.
    - exp(-inf) becomes 0, which is correct for masked lanes.
    - Softmax is reduction-heavy: max reduction + sum reduction + elementwise normalization.
"""

import time
import numpy as np

try:
    import torch
    import triton
    import triton.language as tl
except Exception:
    torch = triton = tl = None


def row_softmax_reference(x: np.ndarray) -> np.ndarray:
    x32 = x.astype(np.float32)
    m = np.max(x32, axis=1, keepdims=True)
    e = np.exp(x32 - m)
    return e / np.sum(e, axis=1, keepdims=True)


def row_softmax_program_simulation(x: np.ndarray, block: int) -> np.ndarray:
    """CPU simulation of one-row-per-program Triton softmax."""
    m, n = x.shape
    out = np.empty((m, n), dtype=np.float32)
    for row in range(m):
        offsets = np.arange(block)
        mask = offsets < n
        safe = np.where(mask, offsets, 0)
        vals = np.where(mask, x[row, safe].astype(np.float32), -np.inf)
        row_max = np.max(vals)
        numerator = np.exp(vals - row_max)
        numerator = np.where(mask, numerator, 0.0)
        denom = np.sum(numerator)
        out[row, :] = numerator[:n] / denom
    return out


if tl is not None:
    @triton.jit
    def row_softmax_kernel(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, stride_xm: tl.constexpr, stride_om: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N

        # Load one row. Invalid columns use -inf so they do not affect max and exp.
        x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=-float('inf')).to(tl.float32)
        x = x - tl.max(x, axis=0)
        numerator = tl.exp(x)
        denominator = tl.sum(numerator, axis=0)
        y = numerator / denominator
        tl.store(out_ptr + row * stride_om + cols, y, mask=mask)


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def row_softmax_triton(x, block: int | None = None):
    if triton is None or torch is None:
        raise RuntimeError('Triton/Torch not available')
    if x.ndim != 2:
        raise ValueError('x must be rank-2 [M, N]')
    M, N = x.shape
    block = block or _next_power_of_2(N)
    out = torch.empty_like(x, dtype=torch.float32)
    row_softmax_kernel[(M,)](x, out, M, N, x.stride(0), out.stride(0), BLOCK=block)
    return out


def gpu_correctness_test(m: int = 129, n: int = 513) -> None:
    if torch is None or triton is None or not torch.cuda.is_available():
        print('GPU/Triton not available; skipping')
        return
    x = torch.randn((m, n), device='cuda', dtype=torch.float32)
    y = row_softmax_triton(x)
    expected = torch.softmax(x.float(), dim=1)
    torch.testing.assert_close(y, expected, rtol=1e-4, atol=1e-4)


def benchmark_stub(m: int = 512, n: int = 1024) -> dict[str, float]:
    x = np.random.default_rng(3).normal(size=(m, n)).astype(np.float32)
    block = _next_power_of_2(n)
    t0 = time.perf_counter(); out = row_softmax_program_simulation(x, block); t1 = time.perf_counter()
    assert np.allclose(out, row_softmax_reference(x), rtol=1e-5, atol=1e-6)
    return {'cpu_seconds': t1 - t0, 'approx_bytes_read_write': float(2 * m * n * 4)}


def smoke_test() -> None:
    x = np.random.default_rng(4).normal(size=(7, 13)).astype(np.float32)
    out = row_softmax_program_simulation(x, block=16)
    assert np.allclose(out, row_softmax_reference(x), rtol=1e-5, atol=1e-6)
    assert np.allclose(np.sum(out, axis=1), 1.0, atol=1e-6)

if __name__ == '__main__':
    print(benchmark_stub())
