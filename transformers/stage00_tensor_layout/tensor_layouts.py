"""
Tensor layout foundations for Transformer kernels.

Beginner idea:
    A tensor shape is not enough. Low-level kernels care about strides, contiguity,
    coalesced memory access, and whether neighboring threads read neighboring addresses.

Typical attention layouts:
    [B, T, H, Dh]  batch, sequence, heads, head_dim
    [B, H, T, Dh]  batch, heads, sequence, head_dim

Production kernels often prefer a layout that makes the dimension loaded by one warp contiguous.
For attention, Dh is usually contiguous because each dot product needs a vector over head_dim.
"""
from __future__ import annotations
import torch


def describe_tensor(x: torch.Tensor) -> dict:
    """Return the metadata a kernel engineer normally asks about first."""
    return {
        'shape': tuple(x.shape),
        'stride': tuple(x.stride()),
        'dtype': str(x.dtype),
        'device': str(x.device),
        'is_contiguous': x.is_contiguous(),
        'numel': x.numel(),
        'element_size_bytes': x.element_size(),
        'storage_bytes_logical': x.numel() * x.element_size(),
    }


def split_heads_bthd(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Convert [B, T, D] to [B, T, H, Dh].

    Beginner:
        This only changes the view if the input is contiguous.

    Intermediate:
        [B, T, H, Dh] has Dh contiguous. That is good for per-head vector loads.

    Advanced:
        Many kernels immediately transpose to [B, H, T, Dh] so every head can be treated
        as an independent matrix multiplication problem.
    """
    b, t, d = x.shape
    if d % num_heads != 0:
        raise ValueError('hidden dimension must be divisible by num_heads')
    return x.view(b, t, num_heads, d // num_heads)


def bthd_to_bhtd(x: torch.Tensor) -> torch.Tensor:
    """
    Convert [B, T, H, Dh] to [B, H, T, Dh].

    Important low-level detail:
        permute changes strides but does not move data. Many PyTorch ops can consume strided tensors,
        but custom kernels often require contiguous blocks for vectorized loads.
    """
    return x.permute(0, 2, 1, 3)


def ensure_contiguous_for_kernel(x: torch.Tensor) -> torch.Tensor:
    """
    Simulate a kernel boundary that requires contiguous memory.

    Production note:
        Calling contiguous() may allocate and copy. A high-performance runtime tries to avoid
        unnecessary layout conversions by choosing layouts consistently across operators.
    """
    return x if x.is_contiguous() else x.contiguous()


def smoke_test() -> None:
    x = torch.randn(2, 4, 16)
    y = bthd_to_bhtd(split_heads_bthd(x, 4))
    assert y.shape == (2, 4, 4, 4)
    assert describe_tensor(y)['numel'] == x.numel()
