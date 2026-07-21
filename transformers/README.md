# Transformer Systems Blueprint

Implementation-first study reference for low-level Transformer runtime work:
NVIDIA TensorRT-LLM/CUDA/CUTLASS/Triton, AMD ROCm, Cerebras, Qualcomm AI Engine,
vLLM-style serving, and compiler/runtime interviews.

This repo intentionally focuses only on Transformer systems implementation.
It does not cover convolution, generic quantization curricula, or non-Transformer ops.

## Design rules

1. Every file is executable Python where possible.
2. Beginner -> intermediate -> advanced -> expert flow is maintained inside each file.
3. Implementations are reference-quality and heavily commented.
4. GPU concepts are represented as faithful CPU/PyTorch simulations when Python cannot directly issue
   CUDA instructions such as `mma.sync`, `ldmatrix`, or `cp.async`.
5. Each implementation includes shape contracts, complexity notes, memory notes, and interview prompts.

## Recommended reading path

```text
stage00_tensor_layout
stage01_attention_basics
stage02_flash_attention
stage03_positional_embeddings
stage04_kv_cache
stage05_paged_attention
stage06_decode_runtime
stage07_parallelism
stage08_moe_runtime
stage09_backward
stage10_performance
stage11_end_to_end_runtime
```

## Quick test

```bash
python run_all_smoke_tests.py
```

All tests are small CPU tests. They validate numerical parity against direct reference implementations.
