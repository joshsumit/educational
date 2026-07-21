"""
FlashDecode reference.

Decode attention has one query token per sequence but many cached K/V tokens.
For long contexts, one CTA may not provide enough parallelism for a single request.

FlashDecode idea:
    Split the KV sequence into chunks.
    Compute partial attention state for each chunk.
    Merge partial states using online-softmax merge.

This is similar to FlashAttention state merging, adapted to one-query decode.
"""
from __future__ import annotations
import math
import numpy as np


def chunk_state(q: np.ndarray, k_chunk: np.ndarray, v_chunk: np.ndarray) -> tuple[float, float, np.ndarray]:
    scores = k_chunk.astype(np.float32) @ q.astype(np.float32) / math.sqrt(q.shape[0])
    m = float(np.max(scores))
    p = np.exp(scores - m)
    l = float(np.sum(p))
    num = p.astype(np.float32) @ v_chunk.astype(np.float32)
    return m, l, num


def merge_states(a: tuple[float, float, np.ndarray], b: tuple[float, float, np.ndarray]) -> tuple[float, float, np.ndarray]:
    m1, l1, n1 = a
    m2, l2, n2 = b
    m = max(m1, m2)
    s1 = math.exp(m1 - m)
    s2 = math.exp(m2 - m)
    l = l1 * s1 + l2 * s2
    num = n1 * s1 + n2 * s2
    return m, l, num


def flashdecode_one_query(q: np.ndarray, k: np.ndarray, v: np.ndarray, chunk_size: int = 256) -> np.ndarray:
    states = []
    for start in range(0, k.shape[0], chunk_size):
        states.append(chunk_state(q, k[start:start+chunk_size], v[start:start+chunk_size]))
    merged = states[0]
    for s in states[1:]:
        merged = merge_states(merged, s)
    _, l, num = merged
    return num / l


def direct_decode(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    scores = k.astype(np.float32) @ q.astype(np.float32) / math.sqrt(q.shape[0])
    p = np.exp(scores - np.max(scores))
    p = p / np.sum(p)
    return p @ v.astype(np.float32)


def smoke_test() -> None:
    rng = np.random.default_rng(5)
    q = rng.normal(size=(32,)).astype(np.float16)
    k = rng.normal(size=(1001,32)).astype(np.float16)
    v = rng.normal(size=(1001,32)).astype(np.float16)
    assert np.allclose(flashdecode_one_query(q,k,v,127), direct_decode(q,k,v), atol=2e-3)
