# Stage 10 and Stage 11 Integration Guide

## What is included

This package adds inference-runtime attention foundations:

```text
prefill vs decode distinction
KV cache layouts
single-query decode reference
online decode reference
Triton decode kernel
multi-head decode reference
KV memory model
block table basics
paged KV layout
paged attention reference
Triton paged attention skeleton
slot mapping
prefix cache simulator
paged memory/fragmentation model
interview notes
```

## What is intentionally deferred

These should come next:

```text
batched decode attention Triton kernel
flattened active token metadata
production paged attention kernel with vectorized block streaming
KV INT8/FP8 quantization
multi-query/group-query attention
continuous batching scheduler
benchmarking decode bandwidth
```

## Keep the CPU references?

Yes. They are essential for metadata correctness. Most paged-attention bugs are not math bugs; they are block-table,
slot-mapping, shape, or layout bugs.

## Next recommended package

```text
Stage 12 — Quantized Matmul and KV Quantization
Stage 13 — Profiling and Benchmarking Discipline
```
