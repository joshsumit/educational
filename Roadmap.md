# Low-Level AI Systems Repo: Stagewise Additions Roadmap

**Prepared for:** Sumit Joshi  
**Goal:** Prepare a single study/review repository for low-level AI roles at **NVIDIA, AMD, Qualcomm, Cerebras, Tenstorrent**, and adjacent compiler/runtime teams.  
**Scope:** What to add next, stage by stage, on top of the current repo containing legacy decoder/paged-attention folders, `ops/`, staged quantization, transformer systems repos, advanced transformer runtime, and Triton basics.

---

## 0. Executive Summary

Your current repo already has a strong foundation:

- Tensor primitives
- Numerical reductions and scans
- Softmax and attention references
- Transformer runtime and KV cache
- Paged attention walkthroughs
- Quantization stages from INT8 to INT4/NF4/FP8
- Advanced transformer simulations: Tensor Core, FlashDecode, paged scheduler, speculative decoding
- Triton from basics to transformer kernels
- Experimental NVIDIA Blackwell/Gluon kernels in `scratch/`

The next missing layer is **not more basic Transformer ops**. The missing layer is:

1. **Compiler/runtime pipeline**
2. **Vendor-specific architecture mapping**
3. **Production serving runtime**
4. **Real Triton advanced kernels**
5. **Profiling and benchmarking discipline**
6. **Training/backward kernels**
7. **Memory layout and data-format depth**
8. **Production MoE runtime**
9. **Cross-vendor interview maps**

The repo should now evolve from:

```text
I know the ops.
```

to:

```text
I understand how ops become kernels, how kernels map to hardware, how runtimes schedule them, and how to optimize them across vendors.
```

---

# Stage 1 — Repo Cleanup And Canonical Structure

## Why this stage matters

The repo has grown organically:

- older decoder and paged-attention folders
- `ops/` as an early implementation area
- staged quantization package
- transformer-focused generated repos
- Triton-focused generated repo
- `scratch/` with advanced NVIDIA Blackwell/Gluon code

Before adding more, make the structure easier to navigate.

## Add / Restructure

```text
legacy_walkthroughs/
├── decoder_shape_trace/
├── paged_attention_v1/
└── paged_batch_attention_v1/

core_ops/
├── tensor_primitives/
├── reductions_scans/
├── softmax/
├── matmul_attention/
├── norms_activations/
└── conv_embedding_pooling/

quantization_staged/
├── stage00_foundations.py
├── stage01_int8_quantization.py
├── ...
└── stage12_end_to_end_pipeline.py

transformer_systems/
├── stage00_tensor_layout/
├── stage01_attention_basics/
├── ...
└── stage11_end_to_end_runtime/

transformer_advanced/
├── stage00_cuda_execution_model/
├── stage01_tensorcore_programming/
├── ...
└── stage09_end_to_end_advanced_runtime/

triton_from_zero_to_transformers/
├── stage00_environment/
├── stage01_programming_model/
├── ...
└── stage10_project_capstones/

experimental_vendor_kernels/
└── nvidia_blackwell_gluon/
```

## Files to add

```text
REPO_MAP.md
WHAT_TO_STUDY_FOR_EACH_COMPANY.md
HOW_TO_RUN_TESTS.md
LEGACY_CODE_GUIDE.md
EXPERIMENTAL_KERNELS_WARNING.md
```

## Cleanup recommendations

### Move old folders

Move:

```text
Decoder/
PagedAttention/
PagedBatchAttention/
```

to:

```text
legacy_walkthroughs/
```

Reason: these are useful for narration and history, but they are no longer the main implementation path.

### Move scratch kernels

Move:

```text
scratch/
```

to:

```text
experimental_vendor_kernels/nvidia_blackwell_gluon/
```

Add a README explaining:

```text
These kernels are experimental NVIDIA Blackwell/Gluon examples.
They may require specific Triton/Gluon/CUDA/Blackwell environments.
They are not part of the portable smoke-test path.
```

---

# Stage 2 — Compiler Runtime Pipeline

## Why this stage matters

This is the biggest missing area for **Qualcomm, Cerebras, Tenstorrent, AMD, NVIDIA compiler/runtime teams**.

Interviewers may ask:

```text
How does a PyTorch graph become kernels?
How are ops fused?
How are layouts propagated?
How does quantization get lowered?
How are buffers allocated?
How are dynamic shapes handled?
How is a kernel selected?
```

Your repo currently has many operator implementations, but it needs the layer that connects:

```text
model graph -> IR -> optimization -> lowering -> memory plan -> kernel launch
```

## Add folder

```text
compiler_runtime/
├── README.md
├── 00_graph_ir_basics.py
├── 01_operator_schema.py
├── 02_shape_inference.py
├── 03_layout_inference.py
├── 04_pattern_matching_fusion.py
├── 05_linear_bias_gelu_fusion.py
├── 06_attention_fusion_pattern.py
├── 07_quantization_lowering.py
├── 08_memory_planner.py
├── 09_static_vs_dynamic_shapes.py
├── 10_kernel_selection.py
├── 11_cost_model.py
├── 12_runtime_execution_plan.py
└── 13_end_to_end_graph_lowering_demo.py
```

## File details

### `00_graph_ir_basics.py`

Implement:

- `Node`
- `TensorValue`
- `Graph`
- topological sorting
- producer/consumer tracking
- simple graph printing

Example patterns:

```text
input -> linear -> bias_add -> gelu -> linear -> output
input -> qkv_projection -> attention -> output_projection
```

### `01_operator_schema.py`

Implement schemas for:

```text
matmul
linear
bias_add
gelu
silu
rmsnorm
softmax
attention
reshape
transpose
quantize
dequantize
```

Each schema should define:

```text
input ranks
output ranks
allowed dtypes
layout constraints
fusibility
```

### `02_shape_inference.py`

Implement shape rules:

```text
matmul: [M,K] x [K,N] -> [M,N]
linear: [B,T,D] x [D,O] -> [B,T,O]
attention: Q[B,H,Tq,Dh], K[B,H,Tk,Dh], V[B,H,Tk,Dv] -> O[B,H,Tq,Dv]
reshape / transpose / concat / split
```

### `03_layout_inference.py`

Track layouts:

```text
BTHD
BHTD
BTD
row-major
column-major
packed-int4
fp8-block-scaled
paged-kv-block
```

Add cost model for layout conversions.

### `04_pattern_matching_fusion.py`

Implement graph pattern matching:

```text
linear + bias
linear + bias + gelu
rmsnorm + residual
qkv projection fusion
attention pipeline fusion
```

### `05_linear_bias_gelu_fusion.py`

Implement an executable fusion example:

```text
x @ w + bias -> gelu -> output
```

Show:

```text
unfused memory traffic
fused memory traffic
temporary tensors avoided
```

### `06_attention_fusion_pattern.py`

Show how this pattern is detected:

```text
QK^T -> scale -> mask -> softmax -> PV
```

Lower it to:

```text
fused_attention_kernel
flash_attention_kernel
paged_attention_kernel
```

### `07_quantization_lowering.py`

Lower fake-quant graph:

```text
dequant(weight_int4, scale) -> matmul -> bias
```

into:

```text
weight_only_int4_matmul_kernel
```

Support:

```text
INT8 activation quantization
INT4 weight-only quantization
FP8 block-scaled matmul
KV cache quantization
```

### `08_memory_planner.py`

Implement:

- liveness interval analysis
- buffer reuse
- peak memory estimate
- activation memory estimate
- temporary tensor removal after fusion

### `09_static_vs_dynamic_shapes.py`

Explain and simulate:

```text
static shapes
symbolic shapes
ragged batch
dynamic sequence length
compile-time specialization
runtime guards
```

### `10_kernel_selection.py`

Given an op and metadata, select:

```text
naive matmul
blocked matmul
tensorcore matmul
triton matmul
flash attention
decode attention
paged attention
quantized matmul
```

### `11_cost_model.py`

Implement simple cost estimates:

```text
FLOPs
bytes moved
arithmetic intensity
launch overhead
temporary memory
expected bottleneck
```

### `12_runtime_execution_plan.py`

Create an execution plan with:

```text
kernel id
input buffers
output buffers
workspace
stream id
estimated cost
```

### `13_end_to_end_graph_lowering_demo.py`

Show:

```text
Transformer block graph
-> infer shapes
-> infer layouts
-> fuse patterns
-> lower quantization
-> plan memory
-> select kernels
-> output execution plan
```

---

# Stage 3 — Vendor Architecture Mapping

## Why this stage matters

Your current advanced low-level coverage is strongest for NVIDIA concepts. The repo needs explicit sections for:

```text
AMD ROCm/CDNA
Qualcomm HTP / Edge AI
Cerebras wafer-scale
Tenstorrent dataflow
```

The goal is not to fully implement each vendor SDK. The goal is to show you understand how the same kernels map differently across accelerator architectures.

## Add folder

```text
vendor_architectures/
├── README.md
├── cross_vendor_mapping/
├── nvidia_cuda_tensor_cores/
├── amd_rocm_cdna/
├── qualcomm_edge_ai/
├── cerebras_wafer_scale/
└── tenstorrent_dataflow/
```

---

## Stage 3A — Cross-Vendor Mapping

```text
vendor_architectures/cross_vendor_mapping/
├── 00_common_kernel_taxonomy.md
├── 01_gemm_across_vendors.py
├── 02_attention_across_vendors.py
├── 03_quantization_across_vendors.py
├── 04_memory_hierarchy_comparison.md
├── 05_execution_model_comparison.md
└── README.md
```

Cover:

```text
SIMT GPU
wavefront GPU
edge NPU
systolic array
spatial dataflow
wafer-scale fabric
tile-based dataflow
```

For each operation, compare:

```text
GEMM
softmax
attention
decode attention
KV cache
quantized matmul
MoE dispatch
```

---

## Stage 3B — NVIDIA CUDA / Tensor Cores

Your repo already has many NVIDIA-like pieces. Make them explicit and organized.

```text
vendor_architectures/nvidia_cuda_tensor_cores/
├── 00_cuda_execution_model.py
├── 01_warp_block_cta_mapping.py
├── 02_shared_memory_and_banks.py
├── 03_tensor_core_mma_sync.py
├── 04_ldmatrix_layout.py
├── 05_cp_async_pipeline.py
├── 06_tma_blackwell_notes.md
├── 07_cutlass_style_gemm.py
├── 08_flashattention_mapping.py
├── 09_paged_attention_mapping.py
├── 10_tensorRT_llm_runtime_map.md
└── README.md
```

Topics:

```text
warp = 32 lanes
CTA/block tiling
shared memory
bank conflicts
MMA instruction shapes
Tensor Core layouts
cp.async
TMA
persistent kernels
FlashAttention
TensorRT-LLM-style runtime
```

---

## Stage 3C — AMD ROCm / CDNA

```text
vendor_architectures/amd_rocm_cdna/
├── 00_amd_execution_model.py
├── 01_wavefront_vs_warp.py
├── 02_lds_memory_model.py
├── 03_matrix_core_mapping.py
├── 04_hip_kernel_translation.py
├── 05_rocm_gemm_tiling.py
├── 06_mi300_memory_hierarchy.py
├── 07_attention_kernel_mapping.py
├── 08_rocm_vs_cuda_cheatsheet.md
└── README.md
```

Topics:

```text
wavefront vs warp
workgroup vs CTA
LDS vs shared memory
Matrix Core vs Tensor Core
ROCm/HIP mapping
Triton on AMD
MI250 / MI300 memory hierarchy
occupancy differences
```

Implementation ideas:

- simulate wavefront lane mapping
- compare wavefront 64 vs warp 32 reductions
- model LDS bank conflicts
- implement ROCm-style GEMM tiling reference
- map attention QK/PV tiles to wavefront groups

---

## Stage 3D — Qualcomm Edge AI / HTP

```text
vendor_architectures/qualcomm_edge_ai/
├── 00_edge_npu_constraints.py
├── 01_fixed_point_arithmetic.py
├── 02_int8_per_channel_conv.py
├── 03_depthwise_pointwise_fusion.py
├── 04_activation_quantization_pipeline.py
├── 05_memory_tiling_for_sram.py
├── 06_operator_partitioning.py
├── 07_cpu_gpu_npu_fallback.py
├── 08_transformer_on_edge_npu.py
├── 09_kv_cache_on_memory_limited_device.py
└── README.md
```

Topics:

```text
fixed-point arithmetic
INT8 / INT16 inference
per-channel weight quantization
activation calibration
SRAM tiling
power-aware scheduling
unsupported op fallback
mobile memory constraints
KV cache compression
```

Implementation ideas:

- fixed-point requantization with integer multiplier/shift
- memory-limited KV cache estimator
- edge-NPU operator partitioning simulator
- fallback planner: NPU vs GPU vs CPU
- transformer-on-edge latency/memory estimator

---

## Stage 3E — Cerebras Wafer-Scale Dataflow

```text
vendor_architectures/cerebras_wafer_scale/
├── 00_wafer_scale_execution_model.py
├── 01_processing_element_grid.py
├── 02_weight_streaming.py
├── 03_activation_streaming.py
├── 04_dataflow_matmul.py
├── 05_dataflow_attention.py
├── 06_static_scheduling.py
├── 07_memory_vs_compute_locality.py
├── 08_pipeline_parallel_on_wafer.py
└── README.md
```

Topics:

```text
spatial dataflow
processing-element grid
compile-time placement
weight streaming
activation routing
static scheduling
communication fabric
on-chip locality
pipeline parallelism
```

Implementation ideas:

- PE-grid matmul simulator
- weight-streaming timeline
- activation-streaming timeline
- static attention schedule
- locality cost model

---

## Stage 3F — Tenstorrent Dataflow

```text
vendor_architectures/tenstorrent_dataflow/
├── 00_tile_based_execution.py
├── 01_core_grid_and_noc.py
├── 02_circular_buffers.py
├── 03_producer_consumer_kernels.py
├── 04_tile_matmul_dataflow.py
├── 05_attention_dataflow.py
├── 06_prefill_decode_mapping.py
├── 07_memory_sharding.py
├── 08_nvidia_vs_tenstorrent.py
└── README.md
```

Topics:

```text
tile-based execution
core grid
NoC communication
circular buffers
producer/consumer split
explicit data movement
matmul dataflow
attention dataflow
memory sharding
```

Implementation ideas:

- tile object abstraction
- circular buffer simulator
- NoC route-cost estimator
- producer/consumer matmul pipeline
- decode-attention dataflow simulation

---

# Stage 4 — Production Serving Runtime

## Why this stage matters

Your paged attention folders are educational and correctly explain allocator sharing and per-sequence state. But production runtimes avoid Python loops and avoid concatenating pages into temporary tensors.

Production runtime topics include:

```text
request lifecycle
continuous batching
chunked prefill
decode token budgets
block table metadata
prefix cache reuse
KV eviction
multi-LoRA serving
speculative decoding integration
latency/throughput metrics
```

## Add folder

```text
serving_runtime_production/
├── README.md
├── 00_request_lifecycle.py
├── 01_waiting_prefill_decode_states.py
├── 02_token_budget_scheduler.py
├── 03_chunked_prefill_scheduler.py
├── 04_decode_batch_builder.py
├── 05_flattened_active_tokens.py
├── 06_block_table_kernel_metadata.py
├── 07_prefix_cache_reuse.py
├── 08_kv_cache_eviction.py
├── 09_priority_scheduling.py
├── 10_multi_lora_serving.py
├── 11_speculative_decode_scheduler.py
├── 12_latency_throughput_metrics.py
└── 13_end_to_end_serving_simulator.py
```

## Key implementations

### Request lifecycle

States:

```text
WAITING
PREFILLING
DECODING
PAUSED
FINISHED
EVICTED
FAILED
```

### Token budget scheduler

Inputs:

```text
max_num_batched_tokens
max_num_sequences
max_prefill_tokens_per_step
max_decode_tokens_per_step
available_kv_blocks
```

Output:

```text
which requests run this tick
which phase each request runs
how many tokens processed
```

### Chunked prefill

Implement:

```text
long prompt split into chunks
chunk scheduling
partial KV cache population
handoff to decode
```

### Flattened active tokens

Instead of looping per sequence:

```text
for seq in active:
    run decode(seq)
```

Build:

```text
active_token_tensor: [TotalActiveTokens, D]
row_to_request_id: [TotalActiveTokens]
row_to_position: [TotalActiveTokens]
```

### Block table metadata

Represent:

```text
block_table: [num_requests, max_blocks_per_request]
sequence_lengths: [num_requests]
slot_mapping: [total_tokens]
```

### Prefix cache reuse

Implement:

```text
hash prompt blocks
reuse physical KV blocks
increment ref counts
copy-on-write when sequence diverges
```

### KV eviction

Policies:

```text
LRU
priority-based
deadline-based
largest-cache-first
prefix-cache-aware
```

### Multi-LoRA serving

Simulate:

```text
base model shared
adapter id per request
batch grouping by adapter
adapter cache
fused LoRA projection
```

### Metrics

Track:

```text
time to first token
inter-token latency
tokens/sec
requests/sec
KV blocks used
fragmentation
scheduler overhead
prefill/decode mix
```

---

# Stage 5 — Real Triton Advanced Kernels

## Why this stage matters

Your Triton repo has a good basics-to-transformer path, but it needs more real `@triton.jit` kernels and CPU parity references.

## Add folder

```text
triton_advanced_kernels/
├── README.md
├── 00_vector_add_triton.py
├── 01_row_softmax_triton.py
├── 02_layernorm_triton.py
├── 03_rmsnorm_triton.py
├── 04_matmul_triton.py
├── 05_matmul_autotuned.py
├── 06_grouped_matmul.py
├── 07_batched_matmul.py
├── 08_fused_bias_gelu.py
├── 09_fused_rmsnorm_residual.py
├── 10_qk_matmul_triton.py
├── 11_flashattention_v1_triton.py
├── 12_decode_attention_triton.py
├── 13_paged_attention_triton_skeleton.py
├── 14_int8_weight_only_matmul_triton.py
├── 15_block_scaled_matmul_triton.py
├── 16_benchmark_harness.py
└── tests/
```

## Kernel progression

### Beginner real kernels

```text
vector add
scale + bias
ReLU
masked load/store
```

### Intermediate real kernels

```text
row softmax
layernorm
rmsnorm
bias + GELU
matmul
matmul autotune
```

### Advanced real kernels

```text
QK matmul
FlashAttention v1-style kernel
single-token decode attention
paged attention skeleton
INT8 weight-only matmul
block-scaled matmul
```

## Each kernel file should include

```text
CPU reference
Triton kernel
wrapper function
shape comments
masking comments
program_id mapping
BLOCK size constants
optional autotune configs
correctness test
benchmark stub
```

---

# Stage 6 — Profiling And Benchmarking

## Why this stage matters

For NVIDIA and AMD roles, being able to implement a kernel is not enough. You must know how to profile it.

## Add folder

```text
profiling_and_benchmarking/
├── README.md
├── 00_benchmark_harness.py
├── 01_latency_vs_throughput.py
├── 02_warmup_and_synchronization.py
├── 03_roofline_for_gemm.py
├── 04_roofline_for_attention.py
├── 05_decode_bandwidth_model.py
├── 06_occupancy_model.py
├── 07_register_pressure_model.py
├── 08_shared_memory_pressure.py
├── 09_tensorcore_utilization.py
├── 10_memory_coalescing_experiments.py
├── 11_quantized_kernel_benchmark.py
├── 12_serving_runtime_metrics.py
└── 13_profile_report_template.md
```

## Cover metrics

```text
latency
throughput
TFLOPs
GB/s
arithmetic intensity
occupancy
register pressure
shared memory usage
L2 hit rate
DRAM bandwidth
tensor core utilization
kernel launch overhead
scheduler overhead
```

## Add benchmark report template

```text
Kernel:
Input shape:
Dtype:
Hardware:
Baseline:
Optimized:
Warmup iterations:
Measurement iterations:
Latency p50:
Latency p95:
Throughput:
FLOPs:
Bytes moved:
Arithmetic intensity:
Likely bottleneck:
Next tuning action:
```

---

# Stage 7 — Training And Backward Kernels

## Why this stage matters

Your repo is currently inference-heavy. That is appropriate for LLM serving, but NVIDIA/AMD/compiler roles may ask about backward passes.

## Add folder

```text
training_kernels/
├── README.md
├── 00_matmul_backward.py
├── 01_layernorm_backward.py
├── 02_rmsnorm_backward.py
├── 03_softmax_backward.py
├── 04_attention_backward_full.py
├── 05_flashattention_backward_reference.py
├── 06_gelu_backward.py
├── 07_swiglu_backward.py
├── 08_embedding_backward_scatter_add.py
├── 09_optimizer_adamw_step.py
├── 10_distributed_gradient_reduce_scatter.py
└── 11_training_memory_estimator.py
```

## Important derivations

### Attention backward

Forward:

```text
S = QK^T / sqrt(Dh)
P = softmax(S)
O = PV
```

Backward:

```text
dV = P^T dO
dP = dO V^T
dS = softmax_backward(P, dP)
dQ = dS K / sqrt(Dh)
dK = dS^T Q / sqrt(Dh)
```

### LayerNorm backward

Cover:

```text
mean gradient
variance gradient
input gradient
gamma gradient
beta gradient
```

### RMSNorm backward

Cover:

```text
rms gradient
input gradient
weight gradient
```

### Embedding backward

Cover:

```text
scatter_add into embedding gradient
atomic update issue
deterministic accumulation issue
```

---

# Stage 8 — Memory Layout And Format Zoo

## Why this stage matters

Hardware interviews often become layout interviews.

They ask:

```text
What is the memory layout?
What does each lane load?
Is it coalesced?
Does transpose materialize?
Can the transpose be fused?
How is INT4 packed?
Where are FP8 scales stored?
How is the KV cache laid out?
```

## Add folder

```text
memory_layouts/
├── README.md
├── 00_contiguous_strides.py
├── 01_nchw_nhwc.py
├── 02_bshd_bhsd_bthd.py
├── 03_kv_cache_layouts.py
├── 04_paged_kv_block_layout.py
├── 05_tensorcore_swizzled_layout.py
├── 06_fp8_scale_layout.py
├── 07_int4_packed_layout.py
├── 08_moe_token_dispatch_layout.py
├── 09_sequence_parallel_layout.py
├── 10_layout_conversion_cost.py
└── 11_layout_decision_tree.md
```

## Must cover

```text
BTD
BTHD
BHTD
BSHD
NCHW
NHWC
row-major
column-major
packed INT4
FP8 scale layout
paged KV block table layout
MoE permuted token layout
sequence-parallel sharded layout
```

---

# Stage 9 — Production MoE Runtime

## Why this stage matters

MoE is now central to large-scale inference/training systems. Your repo has routing basics, but needs production runtime depth.

## Add folder

```text
moe_production_runtime/
├── README.md
├── 00_topk_router.py
├── 01_expert_capacity_and_dropping.py
├── 02_token_permutation.py
├── 03_grouped_gemm_for_experts.py
├── 04_expert_parallel_all_to_all.py
├── 05_load_balancing_loss.py
├── 06_router_aux_loss.py
├── 07_shared_expert_path.py
├── 08_deepseek_style_moe.py
├── 09_mixtral_style_moe.py
├── 10_moe_serving_scheduler.py
├── 11_moe_memory_bandwidth_model.py
└── 12_end_to_end_moe_runtime.py
```

## Must implement

```text
Top-k routing
capacity handling
token dropping
token permutation
expert grouping
grouped GEMM
all-to-all simulation
combine/scatter back
load-balancing loss
router auxiliary loss
shared expert path
serving-time expert batching
```

---

# Stage 10 — Distributed Parallelism Deep Dive

## Why this stage matters

Your repo has ring/tree allreduce and tensor/ring attention simulations, but it needs a more complete distributed Transformer section.

## Add folder

```text
distributed_transformer_runtime/
├── README.md
├── 00_data_parallel.py
├── 01_tensor_parallel_column_row.py
├── 02_sequence_parallel.py
├── 03_context_parallel.py
├── 04_ring_attention.py
├── 05_pipeline_parallel_1f1b.py
├── 06_expert_parallel.py
├── 07_reduce_scatter_all_gather.py
├── 08_all_to_all_moe.py
├── 09_overlap_comm_compute.py
├── 10_memory_timeline.py
└── 11_hybrid_parallel_plan.py
```

## Must cover

```text
Data parallel
Tensor parallel
Pipeline parallel
Sequence parallel
Context parallel
Expert parallel
Ring attention
Reduce-scatter
All-gather
All-to-all
Communication/compute overlap
Hybrid parallel plans
```

---

# Stage 11 — Quantization Deployment And Runtime

## Why this stage matters

Your quantization math is strong. The next missing piece is deployment: how quantization flows into kernels and hardware execution.

## Add folder

```text
quantization_runtime_deployment/
├── README.md
├── 00_quantized_model_format.py
├── 01_weight_only_runtime.py
├── 02_activation_dynamic_quant_runtime.py
├── 03_kv_cache_quantization.py
├── 04_fp8_transformer_runtime.py
├── 05_int4_weight_packing_runtime.py
├── 06_scale_zero_point_memory_layout.py
├── 07_quantized_kernel_selection.py
├── 08_accuracy_latency_tradeoff.py
├── 09_edge_npu_quantization_path.py
└── 10_server_gpu_quantization_path.py
```

## Add focus areas

```text
W8A8
W4A16
W4A8
FP8 training/inference
KV cache INT8/FP8 quantization
per-token activation quantization
per-channel weight quantization
scale memory overhead
packing/unpacking overhead
accuracy/latency Pareto
```

---

# Stage 12 — Company-Specific Interview Maps

## Why this stage matters

You are preparing for multiple hardware/system companies. Create explicit study maps.

## Add folder

```text
company_lens/
├── README.md
├── nvidia_interview_map.md
├── amd_interview_map.md
├── qualcomm_interview_map.md
├── cerebras_interview_map.md
├── tenstorrent_interview_map.md
├── compiler_runtime_interview_map.md
├── kernel_engineer_interview_map.md
├── inference_runtime_interview_map.md
└── study_order_by_role.md
```

## Each file should include

```text
What this company likely cares about
Repo folders to study
Implementation files to practice
Concepts to explain clearly
Likely coding tasks
Likely design questions
Common traps
What to add next
```

---

# Recommended Implementation Order

Do not implement everything randomly. Add in this order.

## Phase 1: Make repo navigable

```text
Stage 1 — Repo cleanup and canonical structure
Stage 12 — Company-specific interview maps
```

Why: makes everything easier to study and explain.

## Phase 2: Add missing systems layer

```text
Stage 2 — Compiler runtime pipeline
Stage 4 — Production serving runtime
```

Why: these connect individual kernels to actual products.

## Phase 3: Add vendor depth

```text
Stage 3 — Vendor architecture mapping
```

Why: this directly targets NVIDIA, AMD, Qualcomm, Cerebras, Tenstorrent.

## Phase 4: Add implementation depth

```text
Stage 5 — Real Triton advanced kernels
Stage 6 — Profiling and benchmarking
```

Why: this moves you from reference implementation to optimization readiness.

## Phase 5: Add advanced runtime depth

```text
Stage 7 — Training/backward kernels
Stage 8 — Memory layouts
Stage 9 — Production MoE runtime
Stage 10 — Distributed Transformer runtime
Stage 11 — Quantization runtime deployment
```

Why: these are advanced areas that differentiate senior candidates.

---

# Highest-Impact Next Batch To Generate

If you want the next downloadable repo update, generate these first:

```text
compiler_runtime/
serving_runtime_production/
vendor_architectures/
company_lens/
```

These four folders will fill the largest conceptual gap in the current repo.

---

# Interview Readiness Checklist

## NVIDIA / TensorRT-LLM / CUDA / Triton

You should be able to explain:

```text
FlashAttention online softmax
Tensor Core MMA tiling
cta/warp/thread mapping
shared memory bank conflicts
cp.async / TMA / double buffering
paged KV cache
continuous batching
TensorRT-LLM-like runtime scheduling
Triton matmul and attention kernels
profiling with roofline thinking
```

## AMD / ROCm

You should be able to explain:

```text
wavefront vs warp
LDS vs shared memory
Matrix Core vs Tensor Core
ROCm/HIP mapping
Triton on AMD
MI300 memory hierarchy
GEMM and attention tiling on AMD
```

## Qualcomm

You should be able to explain:

```text
fixed-point arithmetic
INT8 per-channel quantization
activation calibration
operator partitioning
NPU/GPU/CPU fallback
SRAM tiling
edge Transformer constraints
KV cache memory pressure
```

## Cerebras

You should be able to explain:

```text
wafer-scale execution
processing-element placement
weight streaming
activation streaming
spatial dataflow
static scheduling
communication fabric
locality management
```

## Tenstorrent

You should be able to explain:

```text
tile-based execution
core grid
NoC communication
circular buffers
producer/consumer kernels
explicit data movement
matmul dataflow
attention dataflow
memory sharding
```

## Compiler / Runtime Teams

You should be able to explain:

```text
graph IR
shape inference
layout inference
fusion patterns
quantization lowering
memory planning
kernel selection
cost modeling
static vs dynamic shapes
runtime execution plan
```

---

# Final Recommendation

The repo should now evolve into four pillars:

```text
1. Operators and kernels
2. Compiler/runtime lowering
3. Serving/runtime scheduling
4. Vendor hardware mapping
```

You already have pillar 1 mostly covered. You have started pillar 3 with paged attention and advanced runtime. You need to add pillar 2 and pillar 4 next.

The next best implementation batch is:

```text
compiler_runtime/
serving_runtime_production/
vendor_architectures/
company_lens/
```

After that, add:

```text
triton_advanced_kernels/
profiling_and_benchmarking/
memory_layouts/
moe_production_runtime/
distributed_transformer_runtime/
training_kernels/
quantization_runtime_deployment/
```

This will make the repo much more credible for **Qualcomm, NVIDIA, Cerebras, Tenstorrent, AMD**, and low-level AI compiler/runtime roles.
