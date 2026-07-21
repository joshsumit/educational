# Triton From Zero To Transformers

A complete implementation-first Triton study repo that starts from basics and then builds toward
Transformer kernels.

This repo has two layers:

1. **Executable CPU references** using NumPy. These always run on a normal machine and are used by
   `run_all_smoke_tests.py`.
2. **Real Triton kernel examples** with `@triton.jit`, written as study-ready source files. These are
   guarded so the repo is still usable when Triton/GPU is not installed.

## Why this repo exists

Most Triton examples jump too quickly to matmul or attention. This repo builds the mental model first:

```text
program ids -> block offsets -> masks -> loads/stores -> elementwise kernels -> reductions
-> matmul tiling -> fusions -> autotuning -> debugging -> transformer kernels
```

## Install

CPU-only smoke tests:

```bash
pip install -r requirements.txt
python run_all_smoke_tests.py
```

Optional Triton/GPU execution:

```bash
pip install triton torch
```

Then run individual `*_triton.py` files on a CUDA/ROCm-supported environment.

## Structure

```text
stage00_environment          repo conventions, optional Triton import checks
stage01_programming_model    program_id, grid, block vectors, launch math
stage02_memory_and_masks     pointer arithmetic, boundary masks, coalescing
stage03_elementwise          vector add, affine, ReLU, dropout-style masks
stage04_reductions           sum, max, softmax, layernorm references
stage05_matmul               naive/block/tiled matmul, real Triton matmul skeleton
stage06_fusions              bias+activation, layernorm, softmax fusion
stage07_autotuning           configuration search and benchmarking harness
stage08_debugging            numerical parity, tolerance, race-condition thinking
stage09_transformer_kernels  QK, attention, causal softmax, decode attention
stage10_project_capstones    mini kernel library and learning checklist
```
