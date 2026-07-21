from __future__ import annotations
try:
    import torch
    import triton
    import triton.language as tl
except Exception:
    torch = triton = tl = None

"""Stage 3.4 — Scale + bias Triton kernel.

Formula:
    out[i] = x[i] * scale + bias

This is the smallest useful fusion pattern. Later kernels combine this with activations:

    out = activation(x * scale + bias)

Performance discussion:
    It is still memory-bound. The win comes when scale/bias is fused into a larger producer or consumer kernel.
"""
import time
import numpy as np


def scale_bias_reference(x: np.ndarray, scale: float, bias: float) -> np.ndarray:
    return x * scale + bias


def scale_bias_blocked_reference(x: np.ndarray, scale: float, bias: float, block: int = 1024) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    for pid in range((x.size + block - 1) // block):
        offsets = pid * block + np.arange(block)
        mask = offsets < x.size
        out[offsets[mask]] = x[offsets[mask]].astype(np.float32) * scale + bias
    return out


if tl is not None:
    @triton.jit
    def scale_bias_kernel(x_ptr, out_ptr, n: tl.constexpr, scale: tl.constexpr, bias: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offsets, x * scale + bias, mask=mask)


def scale_bias_triton(x, scale: float, bias: float, block: int = 1024):
    if triton is None or torch is None: raise RuntimeError('Triton/Torch not available')
    out = torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), block),)
    scale_bias_kernel[grid](x, out, x.numel(), scale=scale, bias=bias, BLOCK=block)
    return out


def benchmark_stub(n: int = 1_000_000, block: int = 1024) -> dict[str, float]:
    x = np.random.default_rng(5).normal(size=n).astype(np.float32)
    t0=time.perf_counter(); out=scale_bias_blocked_reference(x, 2.0, 0.5, block); t1=time.perf_counter()
    assert np.allclose(out, scale_bias_reference(x, 2.0, 0.5))
    return {'cpu_seconds': t1-t0, 'approx_bytes_moved': float(2*n*4)}


def smoke_test() -> None:
    x = np.arange(17, dtype=np.float32)
    assert np.allclose(scale_bias_blocked_reference(x, 2.0, 0.5, block=6), scale_bias_reference(x, 2.0, 0.5))

if __name__ == '__main__': print(benchmark_stub())
