from __future__ import annotations
"""Stage 10.1 — KV cache layouts.

A simple contiguous KV cache can be represented as:

    K[num_heads, max_seq, head_dim]
    V[num_heads, max_seq, head_dim]

For a decode kernel, one program often handles one head or one sequence/head pair and streams:

    K[head, 0:seq_len, :]
    V[head, 0:seq_len, :]

Important layout questions:

1. Are head_dim elements contiguous?
2. Is sequence dimension contiguous or strided?
3. Are K and V stored separately or interleaved?
4. Is cache paged or contiguous?
5. Is KV quantized?
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class KVLayout3D:
    num_heads: int
    max_seq: int
    head_dim: int
    stride_h: int
    stride_t: int
    stride_d: int

    def offset(self, head: int, token: int, dim: int) -> int:
        return head * self.stride_h + token * self.stride_t + dim * self.stride_d


def contiguous_h_t_d_layout(num_heads: int, max_seq: int, head_dim: int) -> KVLayout3D:
    return KVLayout3D(num_heads, max_seq, head_dim, stride_h=max_seq * head_dim, stride_t=head_dim, stride_d=1)


def token_major_t_h_d_layout(num_heads: int, max_seq: int, head_dim: int) -> KVLayout3D:
    # logical indexing still offset(head, token, dim), but physical layout is [T,H,D]
    return KVLayout3D(num_heads, max_seq, head_dim, stride_h=head_dim, stride_t=num_heads * head_dim, stride_d=1)


def gather_k_for_head(k_cache_flat: np.ndarray, layout: KVLayout3D, head: int, seq_len: int) -> np.ndarray:
    out = np.empty((seq_len, layout.head_dim), dtype=k_cache_flat.dtype)
    for t in range(seq_len):
        for d in range(layout.head_dim):
            out[t, d] = k_cache_flat[layout.offset(head, t, d)]
    return out


def smoke_test() -> None:
    layout = contiguous_h_t_d_layout(2, 4, 3)
    assert layout.offset(1, 2, 1) == 19
    x = np.arange(2 * 4 * 3, dtype=np.float32)
    gathered = gather_k_for_head(x, layout, head=1, seq_len=2)
    assert gathered.tolist() == [[12.0, 13.0, 14.0], [15.0, 16.0, 17.0]]
    tm = token_major_t_h_d_layout(2, 4, 3)
    assert tm.offset(1, 2, 1) == 16
