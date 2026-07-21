from __future__ import annotations
try:
    import torch
    import triton
    import triton.language as tl
except Exception:
    torch = triton = tl = None

"""Stage 3.3 — GELU Triton kernel.

Approximate GELU:
    0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))

Why this matters:
    GELU is common in Transformer MLPs. In production, it is often fused with bias add or matmul epilogues.
"""
import time
import numpy as np

SQRT_2_OVER_PI = np.float32(np.sqrt(2.0 / np.pi))


def gelu_reference(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return 0.5 * x * (1.0 + np.tanh(SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)))


def gelu_blocked_reference(x: np.ndarray, block: int = 1024) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    for pid in range((x.size + block - 1) // block):
        offsets = pid * block + np.arange(block)
        mask = offsets < x.size
        vals = x[offsets[mask]].astype(np.float32)
        out[offsets[mask]] = gelu_reference(vals)
    return out


if tl is not None:
    @triton.jit
    def gelu_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = 0.5 * x * (1.0 + tl.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
        tl.store(out_ptr + offsets, y, mask=mask)


def gelu_triton(x, block: int = 1024):
    if triton is None or torch is None: raise RuntimeError('Triton/Torch not available')
    out = torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), block),)
    gelu_kernel[grid](x, out, x.numel(), BLOCK=block)
    return out


def benchmark_stub(n: int = 1_000_000, block: int = 1024) -> dict[str, float]:
    x = np.random.default_rng(4).normal(size=n).astype(np.float32)
    t0 = time.perf_counter(); out = gelu_blocked_reference(x, block); t1 = time.perf_counter()
    assert np.allclose(out, gelu_reference(x), atol=1e-6)
    return {'cpu_seconds': t1 - t0, 'approx_bytes_moved': float(2 * n * 4)}


def smoke_test() -> None:
    x = np.linspace(-3, 3, 29, dtype=np.float32)
    assert np.allclose(gelu_blocked_reference(x, block=5), gelu_reference(x), atol=1e-6)

if __name__ == '__main__': print(benchmark_stub())
