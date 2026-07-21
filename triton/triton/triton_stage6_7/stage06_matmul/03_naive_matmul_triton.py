from __future__ import annotations
"""Stage 6.3 — Naive Triton matmul kernel.

This is intentionally simple and not production-quality.

Program mapping:
    One Triton program computes one output element C[m, n].

Why include it?
    - It makes the scalar M/N/K math obvious.
    - It is useful before jumping to tiled matmul.

Why this kernel is bad for performance:
    - Too many programs for large matrices.
    - Each program does only one output element.
    - No reuse of A/B tiles inside a program.
    - Does not use `tl.dot`.

You should understand this kernel, then move on quickly to the tiled version.
"""

import numpy as np

try:
    import torch
    import triton
    import triton.language as tl
except Exception:
    torch = triton = tl = None


def matmul_naive_reference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a.astype(np.float32) @ b.astype(np.float32)


if tl is not None:
    @triton.jit
    def matmul_element_kernel(
        a_ptr, b_ptr, c_ptr,
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        stride_am: tl.constexpr, stride_ak: tl.constexpr,
        stride_bk: tl.constexpr, stride_bn: tl.constexpr,
        stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row = pid // N
        col = pid % N
        acc = tl.full((), 0.0, tl.float32)
        for kk in range(0, K):
            av = tl.load(a_ptr + row * stride_am + kk * stride_ak).to(tl.float32)
            bv = tl.load(b_ptr + kk * stride_bk + col * stride_bn).to(tl.float32)
            acc += av * bv
        tl.store(c_ptr + row * stride_cm + col * stride_cn, acc)


def matmul_naive_triton(a, b):
    if triton is None or torch is None:
        raise RuntimeError('Triton/Torch not available')
    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise ValueError('shape mismatch')
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (M * N,)
    matmul_element_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
    return c


def gpu_correctness_test() -> None:
    if torch is None or triton is None or not torch.cuda.is_available():
        print('GPU/Triton not available; skipping')
        return
    a = torch.randn((13, 17), device='cuda', dtype=torch.float32)
    b = torch.randn((17, 11), device='cuda', dtype=torch.float32)
    y = matmul_naive_triton(a, b)
    torch.testing.assert_close(y, a @ b, rtol=1e-4, atol=1e-4)


def smoke_test() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(size=(7, 9)).astype(np.float32)
    b = rng.normal(size=(9, 5)).astype(np.float32)
    assert np.allclose(matmul_naive_reference(a, b), a @ b, atol=1e-5)

if __name__ == '__main__':
    smoke_test(); gpu_correctness_test()
