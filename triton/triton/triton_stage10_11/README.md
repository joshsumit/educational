# Triton From Zero To Expert — Stage 10 and Stage 11

This is the sixth downloadable implementation wave.

It adds the inference-runtime attention path:

```text
Stage 10 — Decode Attention and KV Cache
Stage 11 — Paged Attention and Block Tables
```

These stages are high ROI for NVIDIA TensorRT-LLM-style runtime interviews, vLLM/SGLang-style serving runtime discussions, and systems-level accelerator interviews.

## Included

```text
stage10_decode_kv_cache/
  00_prefill_vs_decode.py
  01_kv_cache_layouts.py
  02_decode_attention_reference.py
  03_decode_attention_triton.py
  04_multihead_decode_reference.py
  05_kv_cache_memory_model.py
  06_decode_interview_notes.py

stage11_paged_attention/
  00_block_table_basics.py
  01_paged_kv_cache_layout.py
  02_paged_attention_reference.py
  03_paged_attention_triton_skeleton.py
  04_slot_mapping_and_append.py
  05_prefix_cache_simulator.py
  06_paged_attention_memory_model.py
  07_paged_attention_interview_notes.py
```

## Why this matters

Prefill and decode have very different kernel behavior:

```text
prefill: many query tokens, QK/PV matmul-heavy

decode: one new query token per sequence, streams the full K/V cache, often memory-bandwidth bound
```

Paged attention adds the production runtime layer:

```text
logical token positions -> logical pages -> physical KV blocks -> block table -> kernel metadata
```

## Run CPU-safe tests

```bash
pip install -r requirements.txt
python run_all_smoke_tests.py
```

Optional GPU/Triton execution:

```bash
pip install torch triton
python -m stage10_decode_kv_cache.03_decode_attention_triton
python -m stage11_paged_attention.03_paged_attention_triton_skeleton
```
