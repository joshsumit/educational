from __future__ import annotations
"""Stage 13.2 — GPU timer stub for Triton kernels.

This file is guarded. It can be imported CPU-only.

For real GPU timing:
    - use torch.cuda.Event or triton.testing.do_bench
    - synchronize around timings
    - separate compile/autotune time from steady-state time
"""

try:
    import torch
    import triton
except Exception:
    torch = triton = None


def gpu_available() -> bool:
    return bool(torch is not None and hasattr(torch, 'cuda') and torch.cuda.is_available())


def do_bench_if_available(fn, warmup: int = 25, rep: int = 100):
    if triton is None or not gpu_available():
        raise RuntimeError('GPU/Triton not available')
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep)


def smoke_test() -> None:
    assert isinstance(gpu_available(), bool)
