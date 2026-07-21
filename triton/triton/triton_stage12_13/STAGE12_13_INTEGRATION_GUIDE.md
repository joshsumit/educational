# Stage 12 and Stage 13 Integration Guide

## What is included

```text
INT8 quantization reference
INT8 matmul reference
INT8 Triton matmul teaching kernel
INT4 packing/unpacking
weight-only INT4 reference
weight-only INT4 Triton skeleton
KV cache quantization
FP8/block-scaled reference
benchmarking rules
timer harness
GPU timer stub
roofline model
matmul/attention/decode models
profile report template
interview notes
```

## What is intentionally deferred

```text
production INT4 tensor-core kernel
Marlin/AWQ/GPTQ-specific layouts
real hardware FP8 instructions
KV quantized paged attention kernel
Nsight Compute metric parsing
occupancy/register/shared-memory analytical model
```

These are advanced vendor/hardware-specific stages.

## Keep CPU references?

Yes. Quantization bugs are often scale/layout/packing bugs, so CPU references are critical.

## Next recommended package

```text
Stage 14 — Advanced Quantized/Paged Runtime Kernels
Stage 15 — MoE and Grouped GEMM
```
