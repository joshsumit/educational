from __future__ import annotations
"""Stage 4.0 — Sum reduction in Triton.

Goal:
    Compute one scalar sum over a 1D tensor.

Why this is more subtle than vector add:
    Vector add is embarrassingly parallel. Each output element is independent.
    A reduction combines many elements into fewer values. That means multiple programs may produce partial
    sums, and a second stage may combine those partial sums.

This file implements a simple two-level reduction design:

    stage A:
        each Triton program sums BLOCK elements and writes one partial sum

    stage B:
        another kernel or CPU/Torch call reduces partial sums

For teaching, the wrapper uses `torch.sum(partials)` for the second stage. Later, a complete GPU-only reduction
can add a recursive Triton reduction over the partial buffer.

Interview notes:
    - For reductions, invalid lanes should usually load 0 for sum.
    - A sum reduction is bandwidth-sensitive for large tensors.
    - Numerical error depends on accumulation dtype and reduction order.
    - fp32 accumulation is common even when inputs are fp16/bf16.
"""

import time
import numpy as np

try:
    import torch
    import triton
    import triton.language as tl
except Exception:
    torch = triton = tl = None


def sum_reference(x: np.ndarray) -> np.float32:
    """NumPy correctness oracle using fp32 accumulation."""
    return np.float32(np.sum(x.astype(np.float32)))


def sum_blocked_reference(x: np.ndarray, block: int = 1024) -> tuple[np.float32, np.ndarray]:
    """CPU simulation of the two-level Triton reduction.

    Returns:
        total: final reduced value
        partials: one partial sum per simulated program
    """
    partials = []
    for pid in range((x.size + block - 1) // block):
        offsets = pid * block + np.arange(block)
        mask = offsets < x.size
        safe = np.where(mask, offsets, 0)
        vals = np.where(mask, x[safe].astype(np.float32), 0.0)
        partials.append(np.sum(vals, dtype=np.float32))
    partials_np = np.asarray(partials, dtype=np.float32)
    return np.float32(np.sum(partials_np, dtype=np.float32)), partials_np


if tl is not None:
    @triton.jit
    def sum_stage1_kernel(x_ptr, partial_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n

        # For sum, invalid lanes contribute 0.0.
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        # tl.sum reduces the BLOCK vector inside one program to one scalar.
        partial = tl.sum(x, axis=0)
        tl.store(partial_ptr + pid, partial)


def sum_triton(x, block: int = 1024):
    """Run the two-level Triton sum.

    The second stage uses torch.sum for simplicity. This keeps Stage 4 focused on the first reduction pattern.
    """
    if triton is None or torch is None:
        raise RuntimeError('Triton/Torch not available')
    n = x.numel()
    grid = (triton.cdiv(n, block),)
    partial = torch.empty((grid[0],), device=x.device, dtype=torch.float32)
    sum_stage1_kernel[grid](x, partial, n, BLOCK=block)
    return torch.sum(partial)


def gpu_correctness_test(n: int = 100_003, block: int = 1024) -> None:
    """Optional GPU test. Run only where torch+triton+GPU are available."""
    if torch is None or triton is None or not torch.cuda.is_available():
        print('GPU/Triton not available; skipping')
        return
    x = torch.randn((n,), device='cuda', dtype=torch.float32)
    y = sum_triton(x, block)
    expected = torch.sum(x.float())
    torch.testing.assert_close(y, expected, rtol=1e-4, atol=1e-4)


def benchmark_stub(n: int = 1_000_000, block: int = 1024) -> dict[str, float]:
    rng = np.random.default_rng(0)
    x = rng.normal(size=n).astype(np.float32)
    t0 = time.perf_counter(); total, _ = sum_blocked_reference(x, block); t1 = time.perf_counter()
    assert np.allclose(total, sum_reference(x), rtol=1e-5, atol=1e-3)
    return {'cpu_seconds': t1 - t0, 'approx_bytes_read': float(n * 4)}


def smoke_test() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=12345).astype(np.float32)
    total, partials = sum_blocked_reference(x, block=257)
    assert partials.ndim == 1
    assert np.allclose(total, sum_reference(x), rtol=1e-5, atol=1e-3)

if __name__ == '__main__':
    print(benchmark_stub())
