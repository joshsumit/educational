from __future__ import annotations
"""Stage 4.1 — Max reduction in Triton.

Goal:
    Compute max(x) over a 1D tensor.

Important reduction rule:
    For max, invalid lanes should load -inf, not 0.

Why:
    If all valid values are negative and invalid lanes load 0, the result becomes wrong.

Example:
    x = [-10, -7]
    invalid lanes = 0
    max would incorrectly become 0.

This detail appears again in softmax:
    masked scores should become -inf before max and exp.
"""

import time
import numpy as np

try:
    import torch
    import triton
    import triton.language as tl
except Exception:
    torch = triton = tl = None


def max_reference(x: np.ndarray) -> np.float32:
    return np.float32(np.max(x.astype(np.float32)))


def max_blocked_reference(x: np.ndarray, block: int = 1024) -> tuple[np.float32, np.ndarray]:
    partials = []
    for pid in range((x.size + block - 1) // block):
        offsets = pid * block + np.arange(block)
        mask = offsets < x.size
        safe = np.where(mask, offsets, 0)
        vals = np.where(mask, x[safe].astype(np.float32), -np.inf)
        partials.append(np.max(vals))
    partials_np = np.asarray(partials, dtype=np.float32)
    return np.float32(np.max(partials_np)), partials_np


if tl is not None:
    @triton.jit
    def max_stage1_kernel(x_ptr, partial_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf')).to(tl.float32)
        partial = tl.max(x, axis=0)
        tl.store(partial_ptr + pid, partial)


def max_triton(x, block: int = 1024):
    if triton is None or torch is None:
        raise RuntimeError('Triton/Torch not available')
    n = x.numel()
    grid = (triton.cdiv(n, block),)
    partial = torch.empty((grid[0],), device=x.device, dtype=torch.float32)
    max_stage1_kernel[grid](x, partial, n, BLOCK=block)
    return torch.max(partial)


def gpu_correctness_test(n: int = 100_003, block: int = 1024) -> None:
    if torch is None or triton is None or not torch.cuda.is_available():
        print('GPU/Triton not available; skipping')
        return
    x = torch.randn((n,), device='cuda', dtype=torch.float32) - 5.0
    y = max_triton(x, block)
    expected = torch.max(x.float())
    torch.testing.assert_close(y, expected, rtol=0, atol=0)


def benchmark_stub(n: int = 1_000_000, block: int = 1024) -> dict[str, float]:
    x = np.random.default_rng(2).normal(size=n).astype(np.float32)
    t0 = time.perf_counter(); total, _ = max_blocked_reference(x, block); t1 = time.perf_counter()
    assert total == max_reference(x)
    return {'cpu_seconds': t1 - t0, 'approx_bytes_read': float(n * 4)}


def smoke_test() -> None:
    x = -np.arange(1, 1000, dtype=np.float32)
    total, _ = max_blocked_reference(x, block=128)
    assert total == -1.0
    assert total == max_reference(x)

if __name__ == '__main__':
    print(benchmark_stub())
