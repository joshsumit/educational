"""
Advanced serving runtime simulator tying V2 topics together.

Pipeline modeled:
    - requests arrive
    - scheduler admits requests
    - paged KV blocks are allocated
    - FlashDecode-style chunking estimates decode work
    - speculative decoding controls how many target passes are needed
    - completed requests release KV blocks

This is a systems simulator, not a language model.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
from stage06_paged_scheduler.paged_attention_scheduler_deep_dive import BlockAllocator, Request, ContinuousBatchScheduler

@dataclass
class DecodeCost:
    request_id: int
    seq_len: int
    kv_chunks: int
    bytes_read: int


def estimate_flashdecode_cost(request_id: int, seq_len: int, num_kv_heads: int, head_dim: int, chunk_size: int, bytes_per_elem: int = 2) -> DecodeCost:
    chunks = math.ceil(seq_len / chunk_size)
    bytes_read = 2 * seq_len * num_kv_heads * head_dim * bytes_per_elem
    return DecodeCost(request_id, seq_len, chunks, bytes_read)

class AdvancedRuntimeSimulator:
    def __init__(self, num_blocks: int = 128, block_size: int = 16, max_batch: int = 4, num_kv_heads: int = 8, head_dim: int = 128, flashdecode_chunk: int = 256):
        self.allocator = BlockAllocator(num_blocks, block_size)
        self.scheduler = ContinuousBatchScheduler(self.allocator, max_batch)
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.flashdecode_chunk = flashdecode_chunk
        self.cost_trace: list[DecodeCost] = []

    def submit(self, req: Request) -> None:
        self.scheduler.submit(req)

    def run(self) -> list[DecodeCost]:
        step = 0
        while self.scheduler.waiting or self.scheduler.active:
            self.scheduler.admit(step)
            for req in list(self.scheduler.active):
                self.cost_trace.append(
                    estimate_flashdecode_cost(req.request_id, req.total_tokens, self.num_kv_heads, self.head_dim, self.flashdecode_chunk)
                )
            self.scheduler.decode_step(step)
            step += 1
        return self.cost_trace


def smoke_test() -> None:
    rt = AdvancedRuntimeSimulator(num_blocks=64, block_size=8, max_batch=2, num_kv_heads=2, head_dim=16, flashdecode_chunk=32)
    rt.submit(Request(1, prompt_len=10, output_len=3, arrival_step=0))
    rt.submit(Request(2, prompt_len=4, output_len=2, arrival_step=1))
    trace = rt.run()
    assert len(trace) >= 5
    assert all(c.bytes_read > 0 for c in trace)
    assert len(rt.allocator.free) == 64
