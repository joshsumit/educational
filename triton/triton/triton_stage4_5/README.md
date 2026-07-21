# Triton From Zero To Expert — Stage 4 and Stage 5

This is the third downloadable implementation wave.

It adds:

- **Stage 4 — Reductions and Softmax**
- **Stage 5 — Normalization kernels: LayerNorm, RMSNorm, and fused RMSNorm + residual**

These stages are important because reductions are the first place where Triton stops looking like simple
vectorized NumPy and starts looking like real GPU kernel work.

## Why these stages matter

Stage 4 builds the core patterns needed for:

- row-wise reductions
- numerically stable softmax
- masked reductions
- attention score normalization
- FlashAttention online softmax later

Stage 5 builds the kernels commonly found in LLM inference and training runtimes:

- LayerNorm
- RMSNorm
- fused residual + RMSNorm
- memory traffic reasoning
- row-per-program mapping

## Design rule used in every file

Each kernel file includes:

```text
1. Detailed motivation and interview notes
2. NumPy reference implementation
3. CPU blocked/program-style simulation
4. Real @triton.jit kernel guarded by optional imports
5. Python wrapper for GPU execution
6. CPU smoke test
7. GPU test helper where applicable
8. Benchmark stub
9. Memory/performance notes
```

The NumPy versions are not placeholders. Keep them. They are correctness oracles and make the repo runnable on
CPU-only machines.

## Run CPU-safe tests

```bash
pip install -r requirements.txt
python run_all_smoke_tests.py
```

## Optional GPU/Triton execution

```bash
pip install torch triton
python -m stage04_reductions.00_sum_reduction_triton
python -m stage04_reductions.01_max_reduction_triton
python -m stage04_reductions.02_row_softmax_triton
python -m stage05_normalization.00_layernorm_triton
python -m stage05_normalization.01_rmsnorm_triton
python -m stage05_normalization.02_fused_rmsnorm_residual_triton
```

## Integration advice

Copy these folders into your main repo:

```text
stage04_reductions/
stage05_normalization/
```

Do not delete prior NumPy files. Move older placeholder versions to `legacy_notes/` only after these smoke tests
and your GPU tests pass.
