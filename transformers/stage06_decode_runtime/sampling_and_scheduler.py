"""
Sampling and continuous batching scheduler.

A serving runtime repeatedly:
    1. admits new requests
    2. performs prefill for new prompts
    3. performs decode for active requests
    4. samples next tokens
    5. removes completed requests and reclaims KV blocks
"""
from __future__ import annotations
from dataclasses import dataclass, field
import torch


def top_k_sample(logits: torch.Tensor, k: int, temperature: float = 1.0) -> int:
    logits = logits / max(temperature, 1e-6)
    vals, idx = torch.topk(logits, min(k, logits.numel()))
    probs = torch.softmax(vals, dim=0)
    return int(idx[torch.multinomial(probs, 1).item()])


def top_p_sample(logits: torch.Tensor, p: float, temperature: float = 1.0) -> int:
    logits = logits / max(temperature, 1e-6)
    probs = torch.softmax(logits, dim=0)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cdf = torch.cumsum(sorted_probs, dim=0)
    keep = cdf <= p
    keep[0] = True
    kept_probs = sorted_probs[keep]
    kept_probs = kept_probs / kept_probs.sum()
    return int(sorted_idx[keep][torch.multinomial(kept_probs, 1).item()])

@dataclass
class Request:
    request_id: int
    arrival_step: int
    prompt_tokens: list[int]
    max_new_tokens: int
    generated: list[int] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return len(self.generated) >= self.max_new_tokens


def continuous_batch_schedule(requests: list[Request], max_batch: int):
    """Yield active request IDs at each decode step."""
    pending = sorted(requests, key=lambda r: r.arrival_step)
    active: list[Request] = []
    step = 0
    while pending or active:
        while pending and pending[0].arrival_step <= step and len(active) < max_batch:
            active.append(pending.pop(0))
        yield step, [r.request_id for r in active]
        for r in list(active):
            r.generated.append(0)  # placeholder token emitted by scheduler simulation
            if r.done:
                active.remove(r)
        step += 1


def smoke_test() -> None:
    reqs = [Request(1,0,[1,2],2), Request(2,1,[3],1)]
    sched = list(continuous_batch_schedule(reqs, max_batch=2))
    assert sched[0][1] == [1]
    assert 2 in sched[1][1]
