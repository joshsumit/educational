# Stage 8 and Stage 9 Integration Guide

## What is included

This package covers the core attention and FlashAttention progression:

```text
attention shape reasoning
QK^T score reference
causal mask reference
full attention reference
QK Triton kernel
causal softmax Triton kernel
two-pass attention reference
online softmax derivation
online softmax reference
FlashAttention v1-style reference
teaching FlashAttention Triton kernel
memory model
interview notes
```

## What is intentionally deferred

These belong in later stages:

```text
decode attention
paged attention
multi-head fused kernels
FlashAttention backward
dropout attention
varlen/ragged attention
block-sparse attention
Tensor Core layout specialization
production benchmarking/profiling
```

## Keep the two-pass attention files?

Yes.

Even though two-pass attention is not memory efficient, it is the cleanest bridge from:

```text
matmul -> mask -> softmax -> matmul
```

to FlashAttention.

## Next recommended stages

```text
Stage 10 — Decode Attention and KV Cache
Stage 11 — Paged Attention and Block Tables
```

Those are high-ROI for TensorRT-LLM, vLLM, SGLang, NVIDIA inference runtime, and systems interviews.
