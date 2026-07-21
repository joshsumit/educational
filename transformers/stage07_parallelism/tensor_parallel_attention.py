"""
Tensor-parallel attention simulator.

Column-parallel linear:
    Split output features across ranks.
    Each rank computes X @ W_i.
    Outputs are concatenated/all-gathered.

Row-parallel linear:
    Split input features across ranks.
    Each rank computes X_i @ W_i.
    Partial outputs are summed/all-reduced.

This file simulates the math on one process.
"""
from __future__ import annotations
import torch


def column_parallel_linear(x: torch.Tensor, weight: torch.Tensor, world_size: int):
    """weight [Din,Dout], split Dout."""
    chunks = torch.chunk(weight, world_size, dim=1)
    partial = [x @ w for w in chunks]
    return torch.cat(partial, dim=-1), partial


def row_parallel_linear(x: torch.Tensor, weight: torch.Tensor, world_size: int):
    """weight [Din,Dout], split Din and sum partial outputs."""
    x_chunks = torch.chunk(x, world_size, dim=-1)
    w_chunks = torch.chunk(weight, world_size, dim=0)
    partial = [xc @ wc for xc, wc in zip(x_chunks, w_chunks)]
    return sum(partial), partial


def qkv_column_parallel_projection(x: torch.Tensor, w_qkv: torch.Tensor, world_size: int):
    """Simulate Megatron-style column-parallel packed QKV projection."""
    y, shards = column_parallel_linear(x, w_qkv, world_size)
    return y, shards


def smoke_test() -> None:
    x = torch.randn(2, 3, 8)
    w = torch.randn(8, 12)
    y_col, _ = column_parallel_linear(x, w, 3)
    assert torch.allclose(y_col, x @ w)
    y_row, _ = row_parallel_linear(x, w, 2)
    assert torch.allclose(y_row, x @ w, atol=1e-6)
