"""
Speculative decoding runtime.

Goal:
    Use a cheap draft model to propose multiple tokens.
    Use the expensive target model to verify them in one pass.

Acceptance rule in this simplified deterministic version:
    Accept a draft token while it equals target argmax at the same position.
    On first mismatch, append the target token and stop this speculative round.

Production systems use probability-ratio acceptance for stochastic sampling. This file keeps the
control flow explicit and interview-friendly.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import numpy as np

class NextTokenModel(Protocol):
    def logits(self, tokens: list[int]) -> np.ndarray: ...

@dataclass
class TableModel:
    transition: np.ndarray

    def logits(self, tokens: list[int]) -> np.ndarray:
        return self.transition[tokens[-1]]


def argmax_token(logits: np.ndarray) -> int:
    return int(np.argmax(logits))


def draft_tokens(model: NextTokenModel, prefix: list[int], draft_len: int) -> list[int]:
    tokens = list(prefix)
    out = []
    for _ in range(draft_len):
        tok = argmax_token(model.logits(tokens))
        out.append(tok)
        tokens.append(tok)
    return out


def speculative_round(prefix: list[int], draft_model: NextTokenModel, target_model: NextTokenModel, draft_len: int) -> tuple[list[int], int]:
    """Return updated tokens and number of accepted draft tokens."""
    proposed = draft_tokens(draft_model, prefix, draft_len)
    tokens = list(prefix)
    accepted = 0
    for tok in proposed:
        target_tok = argmax_token(target_model.logits(tokens))
        if tok == target_tok:
            tokens.append(tok)
            accepted += 1
        else:
            tokens.append(target_tok)
            return tokens, accepted
    # If all draft tokens accepted, ask target for one extra token to maintain progress.
    tokens.append(argmax_token(target_model.logits(tokens)))
    return tokens, accepted


def speculative_decode(prefix: list[int], draft_model: NextTokenModel, target_model: NextTokenModel, max_new_tokens: int, draft_len: int = 4) -> list[int]:
    tokens = list(prefix)
    while len(tokens) < len(prefix) + max_new_tokens:
        tokens, _ = speculative_round(tokens, draft_model, target_model, draft_len)
    return tokens[:len(prefix) + max_new_tokens]


def smoke_test() -> None:
    vocab = 5
    target = np.zeros((vocab, vocab), dtype=np.float32)
    draft = np.zeros((vocab, vocab), dtype=np.float32)
    for i in range(vocab):
        target[i, (i+1)%vocab] = 1.0
        draft[i, (i+1)%vocab] = 1.0
    out = speculative_decode([0], TableModel(draft), TableModel(target), max_new_tokens=6, draft_len=3)
    assert out == [0,1,2,3,4,0,1]
