from __future__ import annotations
try:
    import torch
    import triton
    import triton.language as tl
except Exception:
    torch = triton = tl = None

"""Stage 3.5 — Fused affine + ReLU Triton kernel.

Formula:
    out[i] = max(x[i] * scale + bias, 0)

Why fuse?
    Unfused path:
        tmp = x * scale + bias      # write tmp to global memory
        out = relu(tmp)             # read tmp from global memory

    Fused path:
        out = relu(x * scale + bias)

    The fused path avoids the temporary global-memory round trip.

This is the first file where the code starts looking like a real production optimization pattern.
"""
import time
import numpy as np


def fused_affine_relu_reference(x: np.ndarray, scale: float, bias: float) -> np.ndarray:
    return np.maximum(x.astype(np.float32) * scale + bias, 0.0)


def fused_affine_relu_blocked_reference(x: np.ndarray, scale: float, bias: float, block: int = 1024) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    for pid in range((x.size + block - 1) // block):
        offsets = pid * block + np.arange(block)
        mask = offsets < x.size
        vals = x[offsets[mask]].astype(np.float32) * scale + bias
        out[offsets[mask]] = np.maximum(vals, 0.0)
    return out


def memory_traffic_bytes(n: int, bytes_per_element: int = 4) -> dict[str, int]:
    """Simple traffic estimate showing why fusion matters.

    Unfused:
        read x, write tmp, read tmp, write out = 4n elements

    Fused:
        read x, write out = 2n elements
    """
    return {
        'unfused_bytes': 4 * n * bytes_per_element,
        'fused_bytes': 2 * n * bytes_per_element,
        'bytes_saved': 2 * n * bytes_per_element,
    }


if tl is not None:
    @triton.jit
    def fused_affine_relu_kernel(x_ptr, out_ptr, n: tl.constexpr, scale: tl.constexpr, bias: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = tl.maximum(x * scale + bias, 0.0)
        tl.store(out_ptr + offsets, y, mask=mask)


def fused_affine_relu_triton(x, scale: float, bias: float, block: int = 1024):
    if triton is None or torch is None: raise RuntimeError('Triton/Torch not available')
    out = torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), block),)
    fused_affine_relu_kernel[grid](x, out, x.numel(), scale=scale, bias=bias, BLOCK=block)
    return out


def benchmark_stub(n: int = 1_000_000, block: int = 1024) -> dict[str, float | int]:
    x = np.random.default_rng(6).normal(size=n).astype(np.float32)
    t0=time.perf_counter(); out=fused_affine_relu_blocked_reference(x, 2.0, -0.1, block); t1=time.perf_counter()
    assert np.allclose(out, fused_affine_relu_reference(x, 2.0, -0.1))
    return {'cpu_seconds': t1-t0, **memory_traffic_bytes(n)}


def smoke_test() -> None:
    x = np.linspace(-2, 2, 37, dtype=np.float32)
    assert np.allclose(fused_affine_relu_blocked_reference(x, 2.0, 0.5, block=8), fused_affine_relu_reference(x, 2.0, 0.5))
    assert memory_traffic_bytes(10)['bytes_saved'] == 80

if __name__ == '__main__': print(benchmark_stub())
