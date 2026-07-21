from __future__ import annotations
"""Stage 11.5 — Prefix cache simulator.

Prefix caching allows multiple requests with the same prompt prefix to share physical KV blocks.

Core idea:
    hash logical prompt block -> physical block
    increase refcount when reused
    copy-on-write when a sequence diverges and needs to modify a shared block

This is a simple metadata simulator, not a full allocator.
"""

from dataclasses import dataclass, field
import hashlib


def hash_token_block(tokens: tuple[int, ...]) -> str:
    h = hashlib.sha256()
    h.update(','.join(map(str, tokens)).encode('utf-8'))
    return h.hexdigest()


@dataclass
class PrefixCache:
    next_physical_block: int = 0
    hash_to_block: dict[str, int] = field(default_factory=dict)
    refcount: dict[int, int] = field(default_factory=dict)

    def get_or_create(self, tokens: tuple[int, ...]) -> int:
        key = hash_token_block(tokens)
        if key in self.hash_to_block:
            block = self.hash_to_block[key]
            self.refcount[block] += 1
            return block
        block = self.next_physical_block
        self.next_physical_block += 1
        self.hash_to_block[key] = block
        self.refcount[block] = 1
        return block

    def release(self, block: int) -> None:
        self.refcount[block] -= 1
        if self.refcount[block] == 0:
            del self.refcount[block]


def smoke_test() -> None:
    cache = PrefixCache()
    b0 = cache.get_or_create((1, 2, 3, 4))
    b1 = cache.get_or_create((1, 2, 3, 4))
    b2 = cache.get_or_create((5, 6, 7, 8))
    assert b0 == b1
    assert b2 != b0
    assert cache.refcount[b0] == 2
    cache.release(b0)
    assert cache.refcount[b0] == 1
