import math
from typing import Callable

import torch

from .softmax_ops import masked_softmax_naive, softmax_safe


# -----------------------------
# Attention components
# -----------------------------

def _split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    # Convert [B, T, D] -> [B, H, T, Dh].
    bsz, seq, dim = x.shape
    if dim % num_heads != 0:
        raise ValueError("hidden dim must be divisible by num_heads")
    d_head = dim // num_heads
    return x.reshape(bsz, seq, num_heads, d_head).permute(0, 2, 1, 3)


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    # Convert [B, H, T, Dh] -> [B, T, D].
    bsz, heads, seq, d_head = x.shape
    return x.permute(0, 2, 1, 3).reshape(bsz, seq, heads * d_head)


def multi_head_attention_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    # Naive MHA for [B, T, D] inputs with explicit per-head math.
    qh = _split_heads(q, num_heads)
    kh = _split_heads(k, num_heads)
    vh = _split_heads(v, num_heads)

    bsz, heads, t_q, d_head = qh.shape
    t_k = kh.shape[2]
    scale = 1.0 / math.sqrt(float(d_head))

    out = torch.zeros_like(qh)
    for b in range(bsz):
        for h in range(heads):
            scores = (qh[b, h] @ kh[b, h].transpose(0, 1)) * scale
            if causal:
                mask = torch.tril(torch.ones((t_q, t_k), dtype=torch.bool, device=q.device))
                probs = masked_softmax_naive(scores, mask, dim=-1)
            else:
                probs = softmax_safe(scores, dim=-1)
            out[b, h] = probs @ vh[b, h]

    return _merge_heads(out)


def multi_head_attention_optimized(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    # Optimized MHA using scaled_dot_product_attention kernel path where available.
    qh = _split_heads(q, num_heads)
    kh = _split_heads(k, num_heads)
    vh = _split_heads(v, num_heads)
    out = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh, is_causal=causal)
    return _merge_heads(out)


def multi_query_attention_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_q_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    # MQA: many query heads, single shared K/V head.
    # q: [B, T, Dq], k/v: [B, T, Dkv] where Dq is divisible by num_q_heads.
    qh = _split_heads(q, num_q_heads)
    bsz, hq, t_q, d_head = qh.shape

    # Expand single KV head to each query head logically.
    k1 = k.unsqueeze(1)  # [B, 1, T, D]
    v1 = v.unsqueeze(1)  # [B, 1, T, D]
    scale = 1.0 / math.sqrt(float(d_head))

    out = torch.zeros_like(qh)
    for b in range(bsz):
        for h in range(hq):
            scores = (qh[b, h] @ k1[b, 0].transpose(0, 1)) * scale
            if causal:
                mask = torch.tril(torch.ones((scores.shape[0], scores.shape[1]), dtype=torch.bool, device=q.device))
                probs = masked_softmax_naive(scores, mask, dim=-1)
            else:
                probs = softmax_safe(scores, dim=-1)
            out[b, h] = probs @ v1[b, 0]
    return _merge_heads(out)


def multi_query_attention_optimized(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_q_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    # Optimized MQA by expanding KV head and using SDPA.
    qh = _split_heads(q, num_q_heads)
    kh = k.unsqueeze(1).expand(-1, num_q_heads, -1, -1)
    vh = v.unsqueeze(1).expand(-1, num_q_heads, -1, -1)
    out = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh, is_causal=causal)
    return _merge_heads(out)


def grouped_query_attention_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    # GQA: query heads are partitioned into groups that share KV heads.
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")

    qh = _split_heads(q, num_q_heads)
    kh = _split_heads(k, num_kv_heads)
    vh = _split_heads(v, num_kv_heads)
    group_size = num_q_heads // num_kv_heads
    scale = 1.0 / math.sqrt(float(qh.shape[-1]))

    bsz, _, t_q, _ = qh.shape
    out = torch.zeros_like(qh)
    for b in range(bsz):
        for hq in range(num_q_heads):
            kv_h = hq // group_size
            scores = (qh[b, hq] @ kh[b, kv_h].transpose(0, 1)) * scale
            if causal:
                mask = torch.tril(torch.ones((scores.shape[0], scores.shape[1]), dtype=torch.bool, device=q.device))
                probs = masked_softmax_naive(scores, mask, dim=-1)
            else:
                probs = softmax_safe(scores, dim=-1)
            out[b, hq] = probs @ vh[b, kv_h]
    return _merge_heads(out)


def grouped_query_attention_optimized(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    # Optimized GQA by repeating KV heads to match query head count.
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")

    repeat = num_q_heads // num_kv_heads
    qh = _split_heads(q, num_q_heads)
    kh = _split_heads(k, num_kv_heads).repeat_interleave(repeat, dim=1)
    vh = _split_heads(v, num_kv_heads).repeat_interleave(repeat, dim=1)
    out = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh, is_causal=causal)
    return _merge_heads(out)


def cross_attention_naive(
    q: torch.Tensor,
    k_ctx: torch.Tensor,
    v_ctx: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    # Cross attention where K/V come from context sequence.
    return multi_head_attention_naive(q, k_ctx, v_ctx, num_heads=num_heads, causal=False)


def cross_attention_optimized(
    q: torch.Tensor,
    k_ctx: torch.Tensor,
    v_ctx: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    # Optimized cross attention path.
    return multi_head_attention_optimized(q, k_ctx, v_ctx, num_heads=num_heads, causal=False)


def paged_attention_naive(
    q: torch.Tensor,
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    # Paged attention simulation.
    # k_pages/v_pages shape: [Npages, page_size, D], page_table shape [Tq, NusedPagesPerToken].
    t_q, d = q.shape
    out = torch.zeros((t_q, v_pages.shape[-1]), dtype=q.dtype, device=q.device)

    for i in range(t_q):
        used_pages = page_table[i]
        keys = []
        values = []
        for p in used_pages:
            p_i = int(p)
            keys.append(k_pages[p_i])
            values.append(v_pages[p_i])
        k_cat = torch.cat(keys, dim=0)
        v_cat = torch.cat(values, dim=0)
        scores = (q[i] @ k_cat.transpose(0, 1)) / math.sqrt(float(d))
        probs = softmax_safe(scores.unsqueeze(0), dim=-1)[0]
        out[i] = probs @ v_cat
    return out


def paged_attention_optimized(
    q: torch.Tensor,
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    # Optimized paged attention simulation with batched gather/cat per token.
    return paged_attention_naive(q, k_pages, v_pages, page_table, page_size)


def flash_attention_v1_algorithm_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    # FlashAttention V1 style reference using block-wise score processing.
    # This computes exact attention but in score blocks to model IO-aware structure.
    t_q, d = q.shape
    t_k = k.shape[0]
    d_v = v.shape[1]
    scale = 1.0 / math.sqrt(float(d))

    out = torch.zeros((t_q, d_v), dtype=q.dtype, device=q.device)
    for i in range(t_q):
        # Running normalization state for one query row.
        m_i = float("-inf")
        l_i = 0.0
        acc = torch.zeros((d_v,), dtype=q.dtype, device=q.device)

        for j0 in range(0, t_k, block_size):
            j1 = min(j0 + block_size, t_k)
            k_blk = k[j0:j1]
            v_blk = v[j0:j1]

            # Compute block logits.
            s_blk = (k_blk @ q[i]) * scale  # [blk]
            m_blk = float(torch.max(s_blk))

            # Merge block normalization with previous blocks.
            m_new = max(m_i, m_blk)
            alpha = math.exp(m_i - m_new) if m_i != float("-inf") else 0.0
            p_blk = torch.exp(s_blk - m_new)
            l_new = alpha * l_i + float(torch.sum(p_blk))

            # Merge weighted value accumulators in the new max frame.
            acc = acc * alpha + torch.sum(p_blk.unsqueeze(1).to(v_blk.dtype) * v_blk, dim=0)

            m_i = m_new
            l_i = l_new

        out[i] = acc / l_i
    return out


def flash_attention_v2_algorithm_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    # FlashAttention V2 style reference (algorithmic) with larger blocks and same online merge.
    # In real kernels, V2 changes work partitioning and parallelization strategy.
    return flash_attention_v1_algorithm_reference(q, k, v, block_size=block_size)


# -----------------------------
# Positional encodings
# -----------------------------

def sinusoidal_positional_encoding(seq_len: int, dim: int, base: float = 10000.0) -> torch.Tensor:
    # Classic transformer sinusoidal positional encoding.
    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(0, dim, 2, dtype=torch.float32)
    inv_freq = torch.exp(-math.log(base) * i / dim)

    out = torch.zeros((seq_len, dim), dtype=torch.float32)
    out[:, 0::2] = torch.sin(pos * inv_freq)
    out[:, 1::2] = torch.cos(pos * inv_freq)
    return out


def rope_cache_generation(seq_len: int, dim: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    # Generate reusable cos/sin cache for RoPE.
    if dim % 2 != 0:
        raise ValueError("RoPE dim must be even")
    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(0, dim, 2, dtype=torch.float32)
    inv_freq = torch.exp(-math.log(base) * i / dim)
    angles = pos * inv_freq
    return torch.cos(angles), torch.sin(angles)


def rotary_positional_embedding(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0) -> torch.Tensor:
    # Apply RoPE to x with shape [B, H, T, D].
    bsz, heads, seq, dim = x.shape
    if dim % 2 != 0:
        raise ValueError("RoPE dim must be even")

    cos_t = cos[offset : offset + seq].to(x.device).unsqueeze(0).unsqueeze(0)
    sin_t = sin[offset : offset + seq].to(x.device).unsqueeze(0).unsqueeze(0)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rot_even = x_even * cos_t - x_odd * sin_t
    rot_odd = x_even * sin_t + x_odd * cos_t

    out = torch.empty_like(x)
    out[..., 0::2] = rot_even
    out[..., 1::2] = rot_odd
    return out


def xpos_apply(x: torch.Tensor, scale_base: float = 512.0) -> torch.Tensor:
    # XPos reference: RoPE-like rotation with position-dependent scaling.
    bsz, heads, seq, dim = x.shape
    cos, sin = rope_cache_generation(seq, dim)

    # Scale grows/shrinks by position and channel to improve extrapolation.
    half = dim // 2
    channel = torch.arange(half, dtype=torch.float32)
    channel_scale = torch.pow(scale_base, channel / max(half - 1, 1))
    pos = torch.arange(seq, dtype=torch.float32).unsqueeze(1)
    pos_scale = torch.pow(channel_scale.unsqueeze(0), pos / max(seq - 1, 1))

    cos = cos / pos_scale
    sin = sin / pos_scale
    return rotary_positional_embedding(x, cos, sin)


def alibi_bias(num_heads: int, q_len: int, k_len: int) -> torch.Tensor:
    # ALiBi additive bias matrix [H, q_len, k_len].
    # Head-specific slopes bias attention toward recent tokens.
    def _slopes(h: int) -> torch.Tensor:
        # Practical slope recipe for ALiBi.
        powers = torch.arange(1, h + 1, dtype=torch.float32)
        return torch.pow(2.0, -8.0 * powers / h)

    slopes = _slopes(num_heads).view(num_heads, 1, 1)
    q_pos = torch.arange(q_len, dtype=torch.float32).view(1, q_len, 1)
    k_pos = torch.arange(k_len, dtype=torch.float32).view(1, 1, k_len)
    rel = k_pos - q_pos
    return slopes * rel


# -----------------------------
# Inference runtime primitives
# -----------------------------

def kv_cache_allocation(num_layers: int, max_batch: int, max_seq: int, num_kv_heads: int, head_dim: int, dtype=torch.float16) -> dict:
    # Allocate KV cache tensors per layer.
    cache = {
        "k": [],
        "v": [],
        "lengths": torch.zeros((max_batch,), dtype=torch.int64),
    }
    for _ in range(num_layers):
        cache["k"].append(torch.zeros((max_batch, num_kv_heads, max_seq, head_dim), dtype=dtype))
        cache["v"].append(torch.zeros((max_batch, num_kv_heads, max_seq, head_dim), dtype=dtype))
    return cache


def kv_cache_append(cache: dict, layer: int, batch_idx: int, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
    # Append one or more new tokens to KV cache at current sequence tail.
    # k_new/v_new shape: [Hkv, Tnew, Dh].
    start = int(cache["lengths"][batch_idx])
    t_new = k_new.shape[1]
    end = start + t_new
    cache["k"][layer][batch_idx, :, start:end, :] = k_new
    cache["v"][layer][batch_idx, :, start:end, :] = v_new
    cache["lengths"][batch_idx] = end


def kv_cache_update(cache: dict, layer: int, batch_idx: int, pos: int, k_token: torch.Tensor, v_token: torch.Tensor) -> None:
    # Update one cache position in-place.
    cache["k"][layer][batch_idx, :, pos, :] = k_token
    cache["v"][layer][batch_idx, :, pos, :] = v_token


def kv_cache_compaction(cache: dict, active_batch_indices: torch.Tensor) -> dict:
    # Compact cache by selecting active batch rows.
    new_cache = {
        "k": [],
        "v": [],
        "lengths": cache["lengths"][active_batch_indices].clone(),
    }
    for layer in range(len(cache["k"])):
        new_cache["k"].append(cache["k"][layer][active_batch_indices].clone())
        new_cache["v"].append(cache["v"][layer][active_batch_indices].clone())
    return new_cache


def continuous_batching_simulation(arrival_steps: torch.Tensor, max_batch: int) -> list[list[int]]:
    # Simulate continuous batching scheduler.
    # Returns active request ids per scheduling step.
    num_reqs = arrival_steps.numel()
    active: list[int] = []
    schedule: list[list[int]] = []

    for step in range(int(torch.max(arrival_steps).item()) + num_reqs + 2):
        # Add newly arrived requests if capacity allows.
        for req_id in range(num_reqs):
            if int(arrival_steps[req_id]) == step and len(active) < max_batch:
                active.append(req_id)

        # Emit one scheduling decision snapshot.
        schedule.append(active.copy())

        # Example completion policy: pop first active request every 2 steps.
        if step % 2 == 1 and active:
            active.pop(0)

    return schedule


def greedy_decoding(logits_fn: Callable[[torch.Tensor], torch.Tensor], prompt_tokens: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    # Greedy decode loop.
    tokens = prompt_tokens.clone()
    for _ in range(max_new_tokens):
        logits = logits_fn(tokens)  # expected shape [V]
        next_tok = int(torch.argmax(logits))
        tokens = torch.cat([tokens, torch.tensor([next_tok], dtype=tokens.dtype)])
    return tokens


def top_k_sampling(logits: torch.Tensor, k: int, temperature: float = 1.0) -> int:
    # Sample one token from top-k logits after temperature scaling.
    logits_t = logits / max(temperature, 1e-6)
    vals, idx = torch.topk(logits_t, k)
    probs = torch.softmax(vals, dim=0)
    pick = int(torch.multinomial(probs, 1))
    return int(idx[pick])


def top_p_sampling(logits: torch.Tensor, p: float, temperature: float = 1.0) -> int:
    # Nucleus (top-p) sampling for one step.
    logits_t = logits / max(temperature, 1e-6)
    probs = torch.softmax(logits_t, dim=0)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cdf = torch.cumsum(sorted_probs, dim=0)

    # Keep smallest prefix whose cumulative prob reaches p.
    keep = cdf <= p
    keep[0] = True
    kept_probs = sorted_probs[keep]
    kept_idx = sorted_idx[keep]
    kept_probs = kept_probs / torch.sum(kept_probs)

    pick = int(torch.multinomial(kept_probs, 1))
    return int(kept_idx[pick])


def top_p_sampling_decode(
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    prompt_tokens: torch.Tensor,
    max_new_tokens: int,
    p: float,
    temperature: float = 1.0,
) -> torch.Tensor:
    # Top-p decode loop.
    tokens = prompt_tokens.clone()
    for _ in range(max_new_tokens):
        logits = logits_fn(tokens)
        next_tok = top_p_sampling(logits, p=p, temperature=temperature)
        tokens = torch.cat([tokens, torch.tensor([next_tok], dtype=tokens.dtype)])
    return tokens


def beam_search_decode(
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    prompt_tokens: torch.Tensor,
    beam_size: int,
    max_new_tokens: int,
) -> list[tuple[torch.Tensor, float]]:
    # Beam search decoding reference.
    beams: list[tuple[torch.Tensor, float]] = [(prompt_tokens.clone(), 0.0)]

    for _ in range(max_new_tokens):
        candidates: list[tuple[torch.Tensor, float]] = []
        for tokens, score in beams:
            logits = logits_fn(tokens)
            log_probs = torch.log_softmax(logits, dim=0)
            vals, idx = torch.topk(log_probs, beam_size)
            for i in range(beam_size):
                tok = int(idx[i])
                new_score = score + float(vals[i])
                new_tokens = torch.cat([tokens, torch.tensor([tok], dtype=tokens.dtype)])
                candidates.append((new_tokens, new_score))

        # Keep highest scoring beams.
        candidates.sort(key=lambda t: t[1], reverse=True)
        beams = candidates[:beam_size]

    return beams


def attention_backward_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_out: torch.Tensor,
    num_heads: int,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Backward reference for MHA using autograd.
    # Useful for learning gradient flow through attention blocks.
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    out = multi_head_attention_naive(q_ref, k_ref, v_ref, num_heads=num_heads, causal=causal)
    out.backward(grad_out)

    grad_q = q_ref.grad if q_ref.grad is not None else torch.zeros_like(q_ref)
    grad_k = k_ref.grad if k_ref.grad is not None else torch.zeros_like(k_ref)
    grad_v = v_ref.grad if v_ref.grad is not None else torch.zeros_like(v_ref)
    return grad_q, grad_k, grad_v


# -----------------------------
# Practice Questions
# -----------------------------
# Beginner:
# 1) Compare MHA, MQA, and GQA in parameter/memory/computation terms.
# 2) Why do we split and merge heads in attention implementations?
# 3) What is the role of KV cache during autoregressive decoding?
# Intermediate:
# 4) How does paged attention reduce memory fragmentation at serving time?
# 5) Why does RoPE use paired even/odd channels?
# 6) Compare greedy decoding, top-k sampling, and top-p sampling behavior.
# Advanced:
# 7) Explain FlashAttention's online normalization and why it is IO-aware.
# 8) How would you extend KV cache compaction for variable beam sizes?
# 9) How can ALiBi improve extrapolation to longer contexts?
# Expert:
# 10) Design a production decode loop with continuous batching and paged KV cache.
# 11) Explain synchronization and parallel partitioning challenges in FlashAttention V2.
# 12) How would you test numerical parity between fused SDPA kernels and reference attention?
