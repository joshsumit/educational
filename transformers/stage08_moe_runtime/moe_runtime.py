"""
Sparse MoE runtime components.

Runtime pipeline:
    1. Router computes logits per token.
    2. Select top-k experts per token.
    3. Dispatch tokens to expert batches.
    4. Run expert MLPs.
    5. Combine outputs using router probabilities.

Distributed production systems add all-to-all communication between dispatch and expert execution.
"""
from __future__ import annotations
import torch


def topk_router(hidden: torch.Tensor, router_weight: torch.Tensor, k: int = 2):
    logits = hidden @ router_weight
    probs = torch.softmax(logits, dim=-1)
    vals, idx = torch.topk(probs, k, dim=-1)
    vals = vals / vals.sum(dim=-1, keepdim=True)
    return idx, vals


def dispatch_tokens(hidden: torch.Tensor, expert_idx: torch.Tensor, num_experts: int):
    """Return token indices assigned to each expert."""
    buckets = {e: [] for e in range(num_experts)}
    for token in range(hidden.shape[0]):
        for slot in range(expert_idx.shape[1]):
            buckets[int(expert_idx[token, slot])].append((token, slot))
    return buckets


def run_moe(hidden: torch.Tensor, router_weight: torch.Tensor, expert_weights: list[tuple[torch.Tensor, torch.Tensor]], k: int = 2):
    idx, gate = topk_router(hidden, router_weight, k)
    out = torch.zeros_like(hidden)
    buckets = dispatch_tokens(hidden, idx, len(expert_weights))
    for expert_id, assignments in buckets.items():
        w1, w2 = expert_weights[expert_id]
        for token, slot in assignments:
            y = torch.relu(hidden[token] @ w1) @ w2
            out[token] += gate[token, slot] * y
    return out, idx, gate


def smoke_test() -> None:
    x = torch.randn(5, 8)
    rw = torch.randn(8, 3)
    experts = [(torch.randn(8,16), torch.randn(16,8)) for _ in range(3)]
    y, idx, gate = run_moe(x, rw, experts, 2)
    assert y.shape == x.shape
    assert idx.shape == (5,2)
