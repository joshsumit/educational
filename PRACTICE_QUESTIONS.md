# Practice Questions

## Phase 1: Numerical Foundations

### Beginner
1. Why is max subtraction required for stable softmax?
2. What is the difference between inclusive and exclusive scan?
3. Why can scatter-add require atomics in parallel implementations?

### Intermediate
1. Compare tree reduction and sequential reduction for numerical error and parallelism.
2. Explain the online softmax update formula and why it is stable.
3. What is the practical difference between reshape and view?

### Advanced
1. Derive fused softmax backward from the Jacobian form.
2. How do segmented scans map to ragged batches?
3. How do gather/scatter access patterns impact memory coalescing?

### Expert
1. Design a block-wise softmax kernel with shared-memory staging.
2. Propose a deterministic reduction strategy across devices.

## Phase 2: Core NN and Attention

### Beginner
1. Derive Conv2D output shape from stride/padding/dilation/kernel.
2. Why is attention score scaling by 1/sqrt(d) used?
3. What is depthwise-separable convolution saving compared to standard conv?

### Intermediate
1. Why does im2col speed compute but increase memory traffic?
2. Compare naive GEMM loop orders for cache locality.
3. Why are embeddings often bandwidth-bound?

### Advanced
1. Explain online attention normalization and memory benefits.
2. How do grouped and depthwise conv affect parameter count and compute?
3. What are transposed-convolution artifact causes?

### Expert
1. Sketch a fused attention kernel (QK, softmax, PV) with tiling.
2. Propose microbenchmarking strategy for GEMM roofline placement.

## Phase 3: Transformer Runtime

### Beginner
1. Compare MHA, MQA, and GQA at a high level.
2. Why does KV cache improve decode throughput?
3. What does causal masking prevent?

### Intermediate
1. Explain paged attention and fragmentation control.
2. Compare greedy, top-k, top-p, and beam search behaviors.
3. How does RoPE differ from absolute positional encoding?

### Advanced
1. Explain FlashAttention V1 online merge math.
2. How does continuous batching improve utilization?
3. What tradeoffs exist between ALiBi and RoPE?

### Expert
1. Design a production decode scheduler with paged KV cache.
2. How would you validate parity between fused SDPA and reference attention?

## Phase 4: Quantization

### Beginner
1. What are scale and zero-point in affine quantization?
2. Why is INT32 accumulation used for INT8 GEMM?
3. Compare symmetric and asymmetric quantization.

### Intermediate
1. Why can per-channel quantization outperform per-tensor quantization?
2. Explain static vs dynamic quantization tradeoffs.
3. Compare INT4 and NF4 at a conceptual level.

### Advanced
1. Explain requantization math in fixed-point pipelines.
2. Compare FP8 E4M3 and E5M2 exponent/mantissa tradeoffs.
3. What failure modes appear with outlier-heavy activation distributions?

### Expert
1. Outline a practical GPTQ or AWQ flow and evaluation criteria.
2. Design a fused quantized linear+bias+activation kernel plan.

## Phase 5-7: GPU/GEMM/Distributed/Sparse/MoE

### Beginner
1. What is coalesced memory access and why does it matter?
2. What causes shared-memory bank conflicts?
3. Explain ring allreduce in simple terms.

### Intermediate
1. How do occupancy limits emerge from registers/shared memory/thread count?
2. Compare ring vs tree allreduce latency/bandwidth behavior.
3. Why can sparse formats reduce memory but not always runtime?

### Advanced
1. Use roofline analysis to classify a kernel bottleneck.
2. Compare output-stationary vs weight-stationary dataflows.
3. Explain expert-capacity handling in MoE and its serving impact.

### Expert
1. Propose communication-compute overlap strategy for distributed training.
2. Design a block-sparse attention or GEMM pipeline with practical constraints.
