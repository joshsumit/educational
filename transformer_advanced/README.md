# Transformer Systems Blueprint V2 Advanced

Advanced implementation-only reference repo for low-level Transformer runtime interviews.

This V2 repo is intentionally separate from V1 and focuses on topics that usually appear in
NVIDIA / AMD / Qualcomm / Cerebras / TensorRT-LLM / Triton / CUTLASS / MLIR-style systems interviews:

- CUDA execution model simulation
- warp/block/thread mapping
- tensor-core `mma.sync` dataflow
- WMMA-style fragments
- `ldmatrix`-style shared-memory loads
- `cp.async` and double buffering
- CUTLASS-style tiled GEMM
- Triton-style attention kernel simulation
- FlashDecode reference
- paged-attention scheduler internals
- speculative decoding runtime
- INT8 / low-bit runtime kernel simulations
- end-to-end advanced serving runtime

## Dependency model

This repo uses only Python standard library plus NumPy.
It does not require PyTorch, CUDA, Triton, or a GPU.

Why NumPy?

The goal is to provide executable algorithmic references for concepts that normally execute inside
GPU kernels. Python cannot issue real instructions such as `mma.sync`, `ldmatrix`, or `cp.async`,
so this repo models their data movement and numerical effect explicitly.

## Run smoke tests

```bash
pip install -r requirements.txt
python run_all_smoke_tests.py
```

## Reading path

```text
stage00_cuda_execution_model
stage01_tensorcore_programming
stage02_async_memory_pipeline
stage03_cutlass_gemm_tiling
stage04_triton_attention_sim
stage05_flashdecode
stage06_paged_scheduler
stage07_speculative_decoding
stage08_quantized_runtime_kernels
stage09_end_to_end_advanced_runtime
```
