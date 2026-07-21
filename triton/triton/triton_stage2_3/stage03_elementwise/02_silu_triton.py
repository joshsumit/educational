from __future__ import annotations
try:
    import torch
    import triton
    import triton.language as tl
except Exception:
    torch = triton = tl = None

"""Stage 3.2 — SiLU / Swish Triton kernel.

Formula:
    silu(x) = x * sigmoid(x)
            = x / (1 + exp(-x))

Why it matters:
    SiLU appears in modern Transformer MLP variants, often in gated forms such as SwiGLU.
"""
import time
import numpy as np


def silu_reference(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def silu_blocked_reference(x: np.ndarray, block: int = 1024) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    for pid in range((x.size + block - 1) // block):
        offsets = pid * block + np.arange(block)
        mask = offsets < x.size
        vals = x[offsets[mask]].astype(np.float32)
        out[offsets[mask]] = vals / (1.0 + np.exp(-vals))
    return out


if tl is not None:
    @triton.jit
    def silu_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = x / (1.0 + tl.exp(-x))
        tl.store(out_ptr + offsets, y, mask=mask)


def silu_triton(x, block: int = 1024):
    if triton is None or torch is None: raise RuntimeError('Triton/Torch not available')
    out = torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), block),)
    silu_kernel[grid](x, out, x.numel(), BLOCK=block)
    return out


def benchmark_stub(n: int = 1_000_000, block: int = 1024) -> dict[str, float]:
    x = np.random.default_rng(3).normal(size=n).astype(np.float32)
    t0 = time.perf_counter(); out = silu_blocked_reference(x, block); t1 = time.perf_counter()
    assert np.allclose(out, silu_reference(x), atol=1e-6)
    return {'cpu_seconds': t1 - t0, 'approx_bytes_moved': float(2 * n * 4)}


def smoke_test() -> None:
    x = np.linspace(-5, 5, 31, dtype=np.float32)
    assert np.allclose(silu_blocked_reference(x, block=7), silu_reference(x), atol=1e-6)

if __name__ == '__main__': print(benchmark_stub())
