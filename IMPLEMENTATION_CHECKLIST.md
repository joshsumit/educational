# Implementation Checklist

## Numerical Foundations

- [ ] softmax_naive_unsafe
- [ ] softmax_safe
- [ ] softmax_online_rowwise
- [ ] masked_softmax_naive
- [ ] masked_softmax_online
- [ ] causal_softmax_naive
- [ ] logsumexp_naive
- [ ] logsumexp_stable
- [ ] softmax_backward_naive
- [ ] softmax_backward_fused
- [ ] reduction_sum_naive
- [ ] reduction_sum_tree
- [ ] reduction_sum_kahan
- [ ] prefix_sum_naive
- [ ] prefix_sum_blelloch_exclusive
- [ ] layernorm_naive
- [ ] layernorm_two_pass
- [ ] rmsnorm_naive
- [ ] rmsnorm_two_pass

## Core NN and Attention

- [ ] conv1d_naive / conv1d_optimized
- [ ] conv2d_naive / conv2d_optimized
- [ ] conv3d_naive / conv3d_optimized
- [ ] depthwise and depthwise-separable conv variants
- [ ] conv_transpose2d_naive / conv_transpose2d_optimized
- [ ] im2col_naive / im2col_optimized
- [ ] col2im_naive / col2im_optimized
- [ ] pooling family (max/avg/global/adaptive)
- [ ] embedding lookup / batched / bag
- [ ] matmul_naive / matmul_tiled
- [ ] attention_naive / attention_online

## Transformer Runtime

- [ ] multi_head_attention_naive / optimized
- [ ] multi_query_attention_naive / optimized
- [ ] grouped_query_attention_naive / optimized
- [ ] cross_attention_naive / optimized
- [ ] paged_attention_naive
- [ ] flash_attention_v1_algorithm_reference
- [ ] flash_attention_v2_algorithm_reference
- [ ] sinusoidal_positional_encoding
- [ ] rope_cache_generation
- [ ] rotary_positional_embedding
- [ ] xpos_apply
- [ ] alibi_bias
- [ ] kv_cache_allocation/append/update/compaction
- [ ] greedy_decoding/top_k_sampling/top_p_sampling/beam_search_decode

## Quantization

- [ ] symmetric/asymmetric INT8 quantization + dequantization
- [ ] per-channel/per-group/per-block quantization
- [ ] INT4/UINT4/INT2/NF4 references
- [ ] FP16/BF16/FP8 references
- [ ] true_int8_gemm_int32_naive + int8_gemm_dequantize
- [ ] fixed_point_matmul_int8
- [ ] weight_only_quantization_reference
- [ ] activation_quantization_reference
- [ ] gptq_style_quantization_reference
- [ ] awq_style_quantization_reference
- [ ] quantize_ste_backward

## GPU, GEMM, Distributed, Sparse, MoE

- [ ] branch_divergence_penalty
- [ ] coalescing_analysis
- [ ] memory_hierarchy_analysis
- [ ] roofline_analysis_matmul
- [ ] compute_occupancy_estimate
- [ ] atomic_add_simulation
- [ ] barrier_synchronization_simulation
- [ ] blocked/register-tiled/packed/batched GEMM
- [ ] ring/tree allreduce, reduce_scatter, all_gather, all_to_all
- [ ] COO/CSR conversion and sparse GEMM variants
- [ ] token routing, top1/top2 routing, capacity handling, dispatch/combine

## Backward Coverage

- [ ] conv2d_backward_reference
- [ ] attention_backward_reference
- [ ] softmax backward variants
- [ ] STE backward references

## Validation

- [ ] python -m compileall ops tests
- [ ] pytest -q tests
- [ ] one stress test per major family (softmax, conv, attention, quant)
