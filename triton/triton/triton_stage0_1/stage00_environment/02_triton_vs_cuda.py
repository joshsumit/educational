"""Stage 0.2 — CUDA vs Triton mental model.

This file is explanatory Python, not a CUDA/Triton runtime file. It gives you exact language to use in
interviews when explaining how Triton differs from CUDA.

CUDA style, simplified:

    __global__ void add(float* x, float* y, float* out, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = x[idx] + y[idx];
        }
    }

Triton style, simplified:

    @triton.jit
    def add_kernel(x, y, out, n: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        xv = tl.load(x + offs, mask=mask, other=0.0)
        yv = tl.load(y + offs, mask=mask, other=0.0)
        tl.store(out + offs, xv + yv, mask=mask)

Main difference:
    CUDA exposes individual threads directly.
    Triton exposes a vectorized block/program abstraction.

Interview-safe phrasing:
    "In Triton, I start by deciding what tile one program owns. Then I express the loads, compute,
    reductions, and stores over vector/tile offsets. The compiler maps that program to lower-level GPU
    execution."
"""
from __future__ import annotations


def cuda_linear_index(block_idx: int, block_dim: int, thread_idx: int) -> int:
    """CUDA-like linear index calculation."""
    return block_idx * block_dim + thread_idx


def triton_offsets(pid: int, block: int) -> list[int]:
    """Triton-like vector offset calculation: pid * BLOCK + arange(0, BLOCK)."""
    return [pid * block + i for i in range(block)]


def compare_cuda_and_triton(block_idx: int, block_dim: int) -> dict[str, object]:
    """Show that CUDA's per-thread indices correspond to Triton's vector of offsets."""
    cuda_indices = [cuda_linear_index(block_idx, block_dim, t) for t in range(block_dim)]
    triton_indices = triton_offsets(block_idx, block_dim)
    return {
        'cuda_indices': cuda_indices,
        'triton_offsets': triton_indices,
        'same_logical_ownership': cuda_indices == triton_indices,
    }


def smoke_test() -> None:
    result = compare_cuda_and_triton(block_idx=2, block_dim=8)
    assert result['same_logical_ownership'] is True
    assert result['triton_offsets'] == [16, 17, 18, 19, 20, 21, 22, 23]
