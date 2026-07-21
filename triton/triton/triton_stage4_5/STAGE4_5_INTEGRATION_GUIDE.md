# Stage 4 and Stage 5 Integration Guide

## Do these files omit anything important?

For this stage, the package intentionally includes the necessary beginner-to-intermediate pieces:

- Sum reduction
- Max reduction
- Stable row-softmax
- Masked/causal softmax
- Reduction interview notes
- LayerNorm
- RMSNorm
- Fused residual + RMSNorm
- Normalization memory model

What is intentionally deferred to later stages:

- full recursive GPU-only reductions
- warp-specialized reductions
- persistent normalization kernels
- dropout + softmax fusion
- FlashAttention online softmax
- normalization backward kernels

Those belong in attention/training/profiling stages, not here.

## Keep NumPy references?

Yes. Keep them.

They are not placeholders. They are the correctness oracle for the real Triton kernels.

Every serious Triton file should have:

```text
NumPy reference
CPU program-style simulation
Triton kernel
wrapper
smoke test
benchmark stub
notes
```

## How to integrate

Copy:

```text
stage04_reductions/
stage05_normalization/
```

into your main Triton repo, then add these modules to your main `run_all_smoke_tests.py`.

## Suggested next stages

Next generate:

```text
Stage 6 — Matmul
Stage 7 — Matmul autotuning and batched matmul
```

That should include naive tiled matmul, grouped matmul order, autotune configs, and batched matmul because those
are prerequisites for QK attention and transformer kernels.
