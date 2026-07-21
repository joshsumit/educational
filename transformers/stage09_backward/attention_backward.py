"""
Attention backward reference and manually decomposed gradients.

Forward:
    S = QK^T / sqrt(D)
    P = softmax(S)
    O = PV

Backward:
    dV = P^T dO
    dP = dO V^T
    dS = softmax_backward(P, dP)
    dQ = dS K / sqrt(D)
    dK = dS^T Q / sqrt(D)
"""
from __future__ import annotations
import math
import torch


def attention_forward_2d(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    s = q @ k.T / math.sqrt(q.shape[-1])
    p = torch.softmax(s, dim=-1)
    o = p @ v
    return o, p, s


def softmax_backward(p: torch.Tensor, dp: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax Jacobian-vector product."""
    return p * (dp - (dp * p).sum(dim=-1, keepdim=True))


def attention_backward_manual(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, grad_o: torch.Tensor):
    o, p, s = attention_forward_2d(q, k, v)
    d_v = p.T @ grad_o
    d_p = grad_o @ v.T
    d_s = softmax_backward(p, d_p)
    scale = 1.0 / math.sqrt(q.shape[-1])
    d_q = d_s @ k * scale
    d_k = d_s.T @ q * scale
    return d_q, d_k, d_v


def smoke_test() -> None:
    q = torch.randn(4, 8, requires_grad=True)
    k = torch.randn(5, 8, requires_grad=True)
    v = torch.randn(5, 8, requires_grad=True)
    go = torch.randn(4, 8)
    o, _, _ = attention_forward_2d(q, k, v)
    o.backward(go)
    dq, dk, dv = attention_backward_manual(q.detach(), k.detach(), v.detach(), go)
    assert torch.allclose(dq, q.grad, atol=1e-6)
    assert torch.allclose(dk, k.grad, atol=1e-6)
    assert torch.allclose(dv, v.grad, atol=1e-6)
