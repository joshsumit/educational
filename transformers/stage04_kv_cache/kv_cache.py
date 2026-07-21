"""
Dense KV cache implementation.

Autoregressive inference stages:
    Prefill: process the prompt and write many K/V tokens into the cache.
    Decode: process one new Q token and attend over cached K/V history.

KV cache shape used here:
    K[layer]: [B, Hkv, MaxSeq, Dh]
    V[layer]: [B, Hkv, MaxSeq, Dh]
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class DenseKVCache:
    num_layers: int
    max_batch: int
    max_seq: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        self.k = [torch.zeros(self.max_batch, self.num_kv_heads, self.max_seq, self.head_dim, dtype=self.dtype) for _ in range(self.num_layers)]
        self.v = [torch.zeros_like(self.k[0]) for _ in range(self.num_layers)]
        self.lengths = torch.zeros(self.max_batch, dtype=torch.long)

    def append(self, layer: int, batch_idx: int, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """Append [Hkv,Tnew,Dh] tensors at the current tail."""
        start = int(self.lengths[batch_idx])
        end = start + k_new.shape[1]
        if end > self.max_seq:
            raise RuntimeError('KV cache capacity exceeded')
        self.k[layer][batch_idx, :, start:end, :] = k_new
        self.v[layer][batch_idx, :, start:end, :] = v_new
        # Length is per request, not per layer. Update once consistently.
        self.lengths[batch_idx] = max(int(self.lengths[batch_idx]), end)

    def view(self, layer: int, batch_idx: int):
        """Return valid K/V history for one request."""
        length = int(self.lengths[batch_idx])
        return self.k[layer][batch_idx, :, :length, :], self.v[layer][batch_idx, :, :length, :]

    def compact(self, active_batch_indices: torch.Tensor) -> 'DenseKVCache':
        """Create a smaller cache containing only active requests."""
        new = DenseKVCache(self.num_layers, active_batch_indices.numel(), self.max_seq, self.num_kv_heads, self.head_dim, self.dtype)
        new.lengths = self.lengths[active_batch_indices].clone()
        for l in range(self.num_layers):
            new.k[l] = self.k[l][active_batch_indices].clone()
            new.v[l] = self.v[l][active_batch_indices].clone()
        return new


def kv_cache_bytes(num_layers: int, batch: int, seq: int, num_kv_heads: int, head_dim: int, bytes_per_elem: int = 2) -> int:
    """K and V both stored, so multiply by 2."""
    return 2 * num_layers * batch * seq * num_kv_heads * head_dim * bytes_per_elem


def smoke_test() -> None:
    c = DenseKVCache(2, 3, 8, 2, 4)
    k = torch.ones(2, 3, 4)
    v = torch.ones(2, 3, 4) * 2
    c.append(0, 1, k, v)
    kk, vv = c.view(0, 1)
    assert kk.shape == (2, 3, 4)
    assert vv.sum().item() == 48
    assert kv_cache_bytes(1,1,1,1,1,2) == 4
