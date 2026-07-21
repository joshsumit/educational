"""
Paged-attention scheduler deep dive.

Serving runtime responsibilities:
    - track waiting requests
    - prefill prompts
    - allocate KV blocks
    - form decode batches
    - reclaim blocks from finished requests
    - optionally reuse prefix-cache blocks

This file implements an executable scheduler model with a block allocator.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque

@dataclass
class Request:
    request_id: int
    prompt_len: int
    output_len: int
    arrival_step: int = 0
    generated: int = 0
    blocks: list[int] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_len + self.generated

    @property
    def done(self) -> bool:
        return self.generated >= self.output_len

class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free = list(range(num_blocks))
        self.ref_count = {i: 0 for i in range(num_blocks)}

    def allocate(self) -> int:
        if not self.free:
            raise RuntimeError('no free KV blocks')
        b = self.free.pop()
        self.ref_count[b] = 1
        return b

    def retain(self, block: int) -> None:
        self.ref_count[block] += 1

    def release(self, block: int) -> None:
        self.ref_count[block] -= 1
        if self.ref_count[block] == 0:
            self.free.append(block)

class ContinuousBatchScheduler:
    def __init__(self, allocator: BlockAllocator, max_batch: int):
        self.allocator = allocator
        self.max_batch = max_batch
        self.waiting: deque[Request] = deque()
        self.active: list[Request] = []
        self.timeline: list[dict] = []

    def submit(self, req: Request) -> None:
        self.waiting.append(req)

    def _ensure_capacity_for_token(self, req: Request) -> None:
        # Allocate a new KV block when token position starts a new block.
        if req.total_tokens % self.allocator.block_size == 0:
            req.blocks.append(self.allocator.allocate())

    def admit(self, step: int) -> None:
        while self.waiting and len(self.active) < self.max_batch and self.waiting[0].arrival_step <= step:
            req = self.waiting.popleft()
            # Allocate prompt blocks.
            needed = (req.prompt_len + self.allocator.block_size - 1) // self.allocator.block_size
            for _ in range(needed):
                req.blocks.append(self.allocator.allocate())
            self.active.append(req)

    def decode_step(self, step: int) -> None:
        self.admit(step)
        batch_ids = [r.request_id for r in self.active]
        self.timeline.append({'step': step, 'active': batch_ids, 'free_blocks': len(self.allocator.free)})
        for req in list(self.active):
            self._ensure_capacity_for_token(req)
            req.generated += 1
            if req.done:
                for b in req.blocks:
                    self.allocator.release(b)
                self.active.remove(req)

    def run_until_done(self, max_steps: int = 1000) -> list[dict]:
        step = 0
        while (self.waiting or self.active) and step < max_steps:
            self.decode_step(step)
            step += 1
        return self.timeline


def smoke_test() -> None:
    alloc = BlockAllocator(num_blocks=20, block_size=4)
    sched = ContinuousBatchScheduler(alloc, max_batch=2)
    sched.submit(Request(1, prompt_len=5, output_len=3, arrival_step=0))
    sched.submit(Request(2, prompt_len=2, output_len=2, arrival_step=1))
    timeline = sched.run_until_done()
    assert timeline[0]['active'] == [1]
    assert any(2 in row['active'] for row in timeline)
    assert len(alloc.free) == 20
