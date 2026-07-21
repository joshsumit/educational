import math
import torch

from .softmax_ops import masked_softmax_naive, softmax_safe


def matmul_naive(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Naive triple-loop matrix multiplication for [M, K] x [K, N].
    m, k_a = a.shape
    k_b, n = b.shape
    if k_a != k_b:
        raise ValueError("Inner dimensions must match for matmul")

    out = torch.zeros((m, n), dtype=a.dtype, device=a.device)
    for i in range(m):
        for j in range(n):
            acc = 0.0
            for k in range(k_a):
                acc += float(a[i, k]) * float(b[k, j])
            out[i, j] = acc
    return out


def matmul_tiled(a: torch.Tensor, b: torch.Tensor, tile_m: int = 32, tile_n: int = 32, tile_k: int = 32) -> torch.Tensor:
    # Tiled matmul reference for [M, K] x [K, N].
    m, k_a = a.shape
    k_b, n = b.shape
    if k_a != k_b:
        raise ValueError("Inner dimensions must match for matmul")

    out = torch.zeros((m, n), dtype=a.dtype, device=a.device)
    for m0 in range(0, m, tile_m):
        m1 = min(m0 + tile_m, m)
        for n0 in range(0, n, tile_n):
            n1 = min(n0 + tile_n, n)
            for k0 in range(0, k_a, tile_k):
                k1 = min(k0 + tile_k, k_a)
                for i in range(m0, m1):
                    for j in range(n0, n1):
                        acc = 0.0
                        for k in range(k0, k1):
                            acc += float(a[i, k]) * float(b[k, j])
                        out[i, j] += acc
    return out


def attention_naive(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    # Naive single-head attention for [T, D], [T, D], [T, Dv].
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(float(d))
    scores = (q @ k.transpose(0, 1)) * scale

    if causal:
        t_q, t_k = scores.shape
        mask = torch.tril(torch.ones((t_q, t_k), dtype=torch.bool, device=scores.device))
        probs = masked_softmax_naive(scores, mask, dim=-1)
    else:
        probs = softmax_safe(scores, dim=-1)
    return probs @ v


def attention_online(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    # Online attention normalization per query row.
    t_q, d = q.shape
    t_k, d_k = k.shape
    if d != d_k:
        raise ValueError("q and k must have the same hidden dimension")

    d_v = v.shape[-1]
    scale = 1.0 / math.sqrt(float(d))
    out = torch.zeros((t_q, d_v), dtype=q.dtype, device=q.device)

    for i in range(t_q):
        running_max = float("-inf")
        running_sum = 0.0
        running_out = torch.zeros((d_v,), dtype=q.dtype, device=q.device)

        for j in range(t_k):
            if causal and j > i:
                continue

            score = float(torch.dot(q[i], k[j]) * scale)
            if score > running_max:
                alpha = math.exp(running_max - score)
                running_out = running_out * alpha + v[j]
                running_sum = running_sum * alpha + 1.0
                running_max = score
            else:
                beta = math.exp(score - running_max)
                running_out = running_out + beta * v[j]
                running_sum += beta

        if running_sum > 0.0:
            out[i] = running_out / running_sum
    return out


    # -----------------------------
    # Practice Questions
    # -----------------------------
    # Beginner:
    # 1) Write the shape equations for matmul_naive and attention_naive inputs/outputs.
    # 2) Why is attention scaled by 1/sqrt(d)?
    # 3) What is the role of causal masking in generation?
    # Intermediate:
    # 4) Compare loop ordering choices in matmul_naive and their cache effects.
    # 5) How does matmul_tiled reduce memory bandwidth pressure?
    # 6) Why can attention_online reduce temporary memory versus full score matrices?
    # Advanced:
    # 7) Derive the online normalization update used in attention_online.
    # 8) How would you fuse QK matmul, softmax, and PV matmul in blocks?
    # 9) What are the precision pitfalls when accumulating in FP16?
    # Expert:
    # 10) Sketch a Tensor Core-ready tiled attention kernel with double buffering.
    # 11) How would you handle variable sequence lengths without wasting compute?
    # 12) Explain how to benchmark arithmetic intensity for attention and GEMM.
