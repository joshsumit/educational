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
- narrated tensor printing with both shape meaning and real values

## Files

- `run_demo.py` - entry point
- `decoder.py` - decoder layer walkthrough
- `attention.py` - attention math with detailed prints
- `paged_kv_cache.py` - paged KV cache implementation
- `page_allocator.py` - physical block allocator
- `tensor_utils.py` - consistent narrated prints, FLOPs, bytes
- `concepts.md` - conceptual explanation

## Run

```bash
python run_demo.py
```

## Shape conventions used everywhere

- `B` = batch size
- `S` = current query sequence length
- `T` = total context tokens already in cache
- `D` = model hidden dimension
- `H` = number of attention heads
- `Dk` = per-head hidden dimension where `Dk = D / H`

Important shape transitions:

1. Input embeddings: `[B, S, D]`
2. After linear Q/K/V projection: `[B, S, D]`
3. After head split: `[B, S, H, Dk]`
4. After transpose for attention kernels: `[B, H, S, Dk]`
5. Gathered cache tensors: `[H, T, Dk]`
6. Attention scores: `[B, H, S, T]`
7. Attention output per head: `[B, H, S, Dk]`
8. Merged heads: `[B, S, D]`

## Why this repo exists

Most educational transformer code explains only math.
This repo also explains **systems behavior**:

- how prefill writes prompt tokens into cache
- how decode reuses old K/V instead of recomputing them
- how pages/blocks are allocated
- why paged attention gathers K/V through a page-table walk
