
# Educational Paged Attention Walkthrough Repo

This repo is a **small, fully-commented educational implementation** of:

- transformer decoder attention
- Q / K / V projection
- head splitting and transpose
- causal masking during prefill
- paged KV cache
- physical block allocation
- logical token -> physical block mapping
- page-table walk during decode
- multi-sequence serving without padding
- educational continuous / inflight batching

## Files

- `run_demo.py` - entry point
- `decoder.py` - decoder layer walkthrough
- `attention.py` - attention math with detailed prints
- `paged_kv_cache.py` - multi-sequence paged KV cache
- `page_allocator.py` - physical block allocator
- `tensor_utils.py` - narrated prints, bytes, simple FLOP helpers
- `concepts.md` - conceptual explanation

## Run

```bash
python run_demo.py
```

## Shape conventions

- `B` = batch size for one dispatched tensor
- `S` = query sequence length for the current call
- `T` = total context tokens for one sequence already in cache
- `D` = model hidden dimension
- `H` = number of attention heads
- `Dk` = per-head hidden dimension where `Dk = D / H`

Important transitions:

1. Input embeddings: `[B, S, D]`
2. After Q/K/V projection: `[B, S, D]`
3. After reshape: `[B, S, H, Dk]`
4. After transpose: `[B, H, S, Dk]`
5. Gathered cache per sequence: `[H, T, Dk]`
6. Attention scores: `[B, H, S, T]`
7. Attention output per head: `[B, H, S, Dk]`
8. Merged heads: `[B, S, D]`



Notes:
1. The Host-Side Iteration Bottleneck
Existing code:  decode_round_no_padding function loops over active sequences using a Python for loop, invoking decode_step independently for each sequence.

The Production Reality: Running sequential host-side loops introduces high CPU tracking overhead. Production engines eliminate this loop by flattening all active tokens into a single execution tensor ([Total_Active_Tokens, D]) and processing them through the linear layers simultaneously.

2. Physical vs. Logical Re-Assembly
Existing code: executes a manual np.concatenate during gather_active_kv to stitch scattered pages back into a uniform tensor right before running attention.

The Production Reality: This step forces the system to perform an intermediate memory copy, which wastes significant memory bandwidth. A production architecture bypasses this duplication completely. It hands a flat Block Table Array metadata descriptor straight down to a specialized CUDA or Triton kernel, letting the hardware stream data directly from non-contiguous pages into local high-speed SRAM caches on the fly.

3:
does not fuse true GPU kernels and does not pack different users into one giant dynamic attention kernel launch but structurally it  mirrors how an inference runtime must think about sequence state, allocator sharing, late arrivals, and no-padding serving.

TODo:
Kernel Fusion & Multi-Head Attention Layouts (FlashAttention / FlashDecoding)
Micro-batching Strategies: Chunked Prefill & Speculative Decoding
Model Quantization at the Compiler Layer (W4A16 vs. KV Cache Quantization)
explicit scheduler class
request lifecycle states: WAITING, PREFILLING, DECODING, FINISHED
continuous batching queue
block reuse after sequence completion
max-tokens budget per round
decode-token budget per scheduler tick
simple free-list reuse when a user finishes