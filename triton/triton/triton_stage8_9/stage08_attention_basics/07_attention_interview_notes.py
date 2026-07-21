from __future__ import annotations
"""Stage 8.7 — Attention interview notes as executable facts."""


def qk_answer() -> str:
    return 'QK^T computes attention scores: Q[Tq,D] times K[Tk,D]^T gives scores[Tq,Tk], scaled by 1/sqrt(D).'


def causal_mask_answer() -> str:
    return 'In causal attention, query row i can attend only to key columns j <= i; masked logits become -inf before softmax.'


def two_pass_limitation_answer() -> str:
    return 'Two-pass attention materializes scores and probabilities, causing O(T^2) memory traffic and memory footprint.'


def attention_flops(t: int, d: int, dv: int | None = None) -> int:
    dv = d if dv is None else dv
    qk = 2 * t * t * d
    pv = 2 * t * t * dv
    return qk + pv


def smoke_test() -> None:
    assert '1/sqrt' in qk_answer()
    assert '-inf' in causal_mask_answer()
    assert attention_flops(4, 8) == 512
