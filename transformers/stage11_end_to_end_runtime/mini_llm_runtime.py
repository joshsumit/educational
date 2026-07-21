"""
Mini end-to-end LLM runtime simulator.

This is not a real language model. It wires together the systems pieces:
    request queue -> paged KV cache -> decode step -> sampling -> completion

The point is to study runtime control flow, not model quality.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import torch
from stage05_paged_attention.paged_kv_cache import PagedKVCache
from stage06_decode_runtime.sampling_and_scheduler import top_k_sample

@dataclass
class RuntimeRequest:
    request_id: int
    prompt: list[int]
    max_new_tokens: int
    generated: list[int] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return len(self.generated) >= self.max_new_tokens

class TinyRuntime:
    def __init__(self, vocab_size: int = 32, hidden: int = 16, num_heads: int = 2, head_dim: int = 8):
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.emb = torch.randn(vocab_size, hidden) * 0.02
        self.lm_head = torch.randn(hidden, vocab_size) * 0.02
        self.cache = PagedKVCache(num_blocks=128, block_size=8, num_kv_heads=num_heads, head_dim=head_dim)

    def prefill(self, req: RuntimeRequest) -> None:
        """Write prompt token embeddings into paged K/V cache as a stand-in for real K/V projections."""
        self.cache.allocate_request(req.request_id)
        for tok in req.prompt:
            h = self.emb[tok]
            kv = h.view(self.num_heads, self.head_dim)
            self.cache.append_token(req.request_id, kv, kv)

    def decode_one(self, req: RuntimeRequest) -> int:
        """Produce one token. Uses mean cached value as a tiny fake decoder state."""
        k, v = self.cache.gather(req.request_id)
        state = v.mean(dim=(0,1))  # [Dh]
        # Expand to hidden by repeating across heads.
        hidden = state.repeat(self.num_heads)
        logits = hidden @ self.lm_head
        next_tok = top_k_sample(logits, k=5)
        req.generated.append(next_tok)
        kv = self.emb[next_tok].view(self.num_heads, self.head_dim)
        self.cache.append_token(req.request_id, kv, kv)
        return next_tok

    def run(self, requests: list[RuntimeRequest]) -> dict[int, list[int]]:
        for r in requests:
            self.prefill(r)
        active = list(requests)
        while active:
            for r in list(active):
                self.decode_one(r)
                if r.done:
                    self.cache.free_request(r.request_id)
                    active.remove(r)
        return {r.request_id: r.prompt + r.generated for r in requests}


def smoke_test() -> None:
    torch.manual_seed(0)
    rt = TinyRuntime()
    out = rt.run([RuntimeRequest(1, [1,2,3], 2), RuntimeRequest(2, [4], 1)])
    assert len(out[1]) == 5
    assert len(out[2]) == 2
