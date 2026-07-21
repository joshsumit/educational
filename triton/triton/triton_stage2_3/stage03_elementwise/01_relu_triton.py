from __future__ import annotations
try:
    import torch
    import triton
    import triton.language as tl
except Exception:
    torch = triton = tl = None

"""Stage 3.1 — ReLU Triton kernel.

Formula:
    out[i] = max(x[i], 0)

Interview points:
    - ReLU is memory-bound: one read and one write per element, very little math.
    - It is often fused into a producer/consumer kernel to avoid extra global-memory traffic.
"""
import time
import numpy as np


def relu_reference(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def relu_blocked_reference(x: np.ndarray, block: int = 1024) -> np.ndarray:
    out = np.empty_like(x)
    for pid in range((x.size + block - 1) // block):
        offsets = pid * block + np.arange(block)
        mask = offsets < x.size
        out[offsets[mask]] = np.maximum(x[offsets[mask]], 0)
    return out


if tl is not None:
    @triton.jit
    def relu_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.maximum(x, 0.0)
        tl.store(out_ptr + offsets, y, mask=mask)


def relu_triton(x, block: int = 1024):
    if triton is None or torch is None:
        raise RuntimeError('Triton/Torch not available')
    out = torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), block),)
    relu_kernel[grid](x, out, x.numel(), BLOCK=block)
    return out


def benchmark_stub(n: int = 1_000_000, block: int = 1024) -> dict[str, float]:
    x = np.random.default_rng(2).normal(size=n).astype(np.float32)
    t0 = time.perf_counter(); out = relu_blocked_reference(x, block); t1 = time.perf_counter()
    assert np.allclose(out, relu_reference(x))
    return {'cpu_seconds': t1 - t0, 'approx_bytes_moved': float(2 * n * 4)}


def smoke_test() -> None:
    x = np.array([-2, -0.5, 0, 1, 3], dtype=np.float32)
    assert np.allclose(relu_blocked_reference(x, block=3), relu_reference(x))

if __name__ == '__main__': print(benchmark_stub())
