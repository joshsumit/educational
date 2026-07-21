# Stage 6 and Stage 7 Integration Guide

## What is included and why

This package covers the essential matmul progression:

```text
shape/stride reasoning
naive matmul reference
tiled CPU matmul
tiled Triton matmul with tl.dot
grouped program order
autotune configuration model
@triton.autotune example
batched matmul reference
batched Triton matmul
precision/accumulation notes
benchmark harness
```

## What is intentionally deferred

These are useful, but belong in later stages:

```text
Tensor Core instruction-level layout details
fp8 matmul
int8 matmul
int4 weight-only matmul
split-K matmul
persistent matmul
grouped GEMM for MoE
QK attention matmul specialization
```

## Keep NumPy versions?

Yes. Keep them.

They serve as:

```text
correctness oracle
shape documentation
CPU smoke tests
interview explanation
debugging baseline
```

## How to merge into your repo

Copy these folders:

```text
stage06_matmul/
stage07_autotuning_batched/
```

Then update your top-level smoke test file with the modules from this repo's `run_all_smoke_tests.py`.

## Next stages

Recommended next package:

```text
Stage 8 — Transformer Attention Basics
Stage 9 — Online Softmax and FlashAttention v1
```

That should include QK matmul, causal mask, attention reference, attention Triton, online softmax, and a teaching
FlashAttention v1 kernel.
