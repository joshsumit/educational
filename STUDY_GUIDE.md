# Low-Level AI Systems Programming Study Guide

## Scope

This is a roadmap-only guide. Implementation code belongs in ops/*.py.

Supplemental one-off scripts are kept in scratch/ and are not the canonical study source.

## Mastery Ladder (Use for Every Topic)

1. Implement by hand.
2. Explain mathematically.
3. Analyze complexity and memory traffic.
4. Map to GPU/kernel/runtime design.

## Validation Gate (Per Phase)

1. Compare with PyTorch reference on small random inputs.
2. Run one numerical-stability stress case.
3. Compute FLOPs, bytes moved, arithmetic intensity.
4. State memory-bound vs compute-bound expectation.
5. Answer practice questions for the phase.

## Phase 1: Numerical Foundations

Files:
- ops/softmax_ops.py
- ops/reduction_scan_ops.py
- ops/norm_activation_ops.py
- ops/tier2_tensor_primitives_ops.py

Targets:
- Stable/online softmax, logsumexp, softmax backward
- Reductions, scans, segmented scans
- LayerNorm, RMSNorm, GELU, SiLU
- Gather/scatter/scatter-add, top-k, sorting, shape transforms

## Phase 2: Core NN Operators

Files:
- ops/tier1_core_nn_ops.py
- ops/matmul_attention_ops.py
- ops/quantization_ops.py

Targets:
- Conv/pool/embedding operator family
- Naive and tiled GEMM
- Single-head attention and online attention
- Basic affine quantization and INT8 reference flow

## Phase 3: Transformer Runtime

Files:
- ops/tier3_transformer_runtime_ops.py

Targets:
- MHA/MQA/GQA/cross/paged attention
- FlashAttention algorithm references
- RoPE/ALiBi/XPos/sinusoidal encoding
- KV cache operations and decode strategies

## Phase 4: Quantization Systems

Files:
- ops/quantization_ops.py
- ops/tier4_quantization_ops.py

Targets:
- Symmetric/asymmetric INT8, static/dynamic
- Per-channel/per-group/per-block
- INT4/UINT4/INT2/NF4
- FP16/BF16/FP8 references
- INT8 GEMM INT32 accumulation and requantization
- STE-style backward references

## Phase 5: GPU Kernel Building Blocks

Files:
- ops/tier5_6_7_numerical_gpu_gemm_ops.py

Targets:
- Coalescing, divergence, bank conflict intuition
- Atomic and barrier simulation
- Roofline and occupancy estimates
- Histogram, transpose, reduction/scan kernels

## Phase 6: Advanced GEMM and Dataflow

Files:
- ops/tier5_6_7_numerical_gpu_gemm_ops.py

Targets:
- Blocked/register-tiled/packed GEMM
- Batched and sparse GEMM variants
- Tensor-core-style and systolic-style simulations
- Output/weight/row stationary dataflows

## Phase 7: Distributed, Sparse, and MoE

Files:
- ops/tier8_9_10_11_distributed_sparse_moe_activation_ops.py

Targets:
- Ring/tree allreduce, reduce-scatter, allgather, all-to-all
- Data/tensor/pipeline parallel simulations
- COO/CSR and sparse GEMM
- MoE routing, capacity handling, dispatch/combine

## Phase 8: Training Fundamentals

Files:
- ops/softmax_ops.py
- ops/tier1_core_nn_ops.py
- ops/tier3_transformer_runtime_ops.py
- ops/tier4_quantization_ops.py

Targets:
- Backward references: softmax, conv2d, attention, STE quantization
- Gradient accumulation and numerical stability checks

## Phase 9: Compiler and Runtime Design

Study topics:
- Operator fusion
- Kernel launch overhead
- Graph capture
- Static vs dynamic shapes
- Layout propagation and memory planning
- Quantization lowering
- Triton/CUDA/CUTLASS/ROCm mental models
- MLIR/XLA-style lowering

## Company Lens

- NVIDIA/AMD: occupancy, coalescing, shared memory, tiling, Tensor Core style mapping
- Qualcomm/edge NPUs: quantization, fixed-point paths, memory bandwidth and power constraints
- Compiler/runtime roles: graph lowering, fusion, dynamic-shape execution tradeoffs
