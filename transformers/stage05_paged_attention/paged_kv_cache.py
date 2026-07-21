"""
Paged KV cache: vLLM-style virtual memory idea implemented as a small reference.

Dense cache problem:
    Reserving [max_batch, max_seq] for every request wastes memory and fragments capacity.

Paged cache idea:
    A request owns logical blocks 0..N.
    Each logical block maps to a physical block from a global pool.
    The block table is the page table.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import torch

@dataclass
class PagedKVCache:
    num_blocks: int
    block_size: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype = torch.float32
    k_blocks: torch.Tensor = field(init=False)
    v_blocks: torch.Tensor = field(init=False)
    free_blocks: list[int] = field(init=False)
    block_tables: dict[int, list[int]] = field(default_factory=dict)
    lengths: dict[int, int] = field(default_factory=dict)
    ref_count: dict[int, int] = field(init=False)

    def __post_init__(self):
        self.k_blocks = torch.zeros(self.num_blocks, self.num_kv_heads, self.block_size, self.head_dim, dtype=self.dtype)
        self.v_blocks = torch.zeros_like(self.k_blocks)
        self.free_blocks = list(range(self.num_blocks))
        self.ref_count = {i: 0 for i in range(self.num_blocks)}

    def _alloc_block(self) -> int:
        if not self.free_blocks:
            raise RuntimeError('out of physical KV blocks')
        b = self.free_blocks.pop()
        self.ref_count[b] = 1
        return b

    def allocate_request(self, request_id: int) -> None:
        self.block_tables[request_id] = []
        self.lengths[request_id] = 0

    def append_token(self, request_id: int, k_token: torch.Tensor, v_token: torch.Tensor) -> None:
        """Append one token [Hkv,Dh] to a request."""
        if request_id not in self.block_tables:
            self.allocate_request(request_id)
        pos = self.lengths[request_id]
        logical_block = pos // self.block_size
        offset = pos % self.block_size
        if logical_block == len(self.block_tables[request_id]):
            self.block_tables[request_id].append(self._alloc_block())
        physical = self.block_tables[request_id][logical_block]
        self.k_blocks[physical, :, offset, :] = k_token
        self.v_blocks[physical, :, offset, :] = v_token
        self.lengths[request_id] += 1

    def gather(self, request_id: int):
        """Return contiguous logical K/V history as [Hkv,T,Dh]."""
        length = self.lengths[request_id]
        ks, vs = [], []
        remaining = length
        for physical in self.block_tables[request_id]:
            take = min(self.block_size, remaining)
            ks.append(self.k_blocks[physical, :, :take, :])
            vs.append(self.v_blocks[physical, :, :take, :])
            remaining -= take
            if remaining <= 0:
                break
        return torch.cat(ks, dim=1), torch.cat(vs, dim=1)

    def fork_prefix(self, src_request: int, dst_request: int) -> None:
        """Share physical blocks for prefix-cache style reuse."""
        self.block_tables[dst_request] = list(self.block_tables[src_request])
        self.lengths[dst_request] = self.lengths[src_request]
        for b in self.block_tables[dst_request]:
            self.ref_count[b] += 1

    def free_request(self, request_id: int) -> None:
        for b in self.block_tables.pop(request_id, []):
            self.ref_count[b] -= 1
            if self.ref_count[b] == 0:
                self.free_blocks.append(b)
        self.lengths.pop(request_id, None)


def smoke_test() -> None:
    c = PagedKVCache(4, 2, 1, 3)
    for i in range(3):
        c.append_token(10, torch.ones(1,3)*i, torch.ones(1,3)*(i+10))
    k, v = c.gather(10)
    assert k.shape == (1,3,3)
    c.fork_prefix(10, 11)
    assert c.ref_count[c.block_tables[10][0]] == 2
    c.free_request(10)
    c.free_request(11)
    assert len(c.free_blocks) == 4
