# Triton From Zero To Expert — Stage 6 and Stage 7

This is the fourth downloadable implementation wave.

It adds the matmul path that later attention, QK, PV, MLP, grouped GEMM, and quantized matmul stages will build on.

## Included stages

```text
stage06_matmul/
  00_matmul_shapes_and_strides.py
  01_naive_matmul_reference.py
  02_tiled_matmul_reference.py
  03_naive_matmul_triton.py
  04_tiled_matmul_triton.py
  05_grouped_matmul_triton.py
  06_matmul_interview_notes.py

stage07_autotuning_batched/
  00_autotune_config_model.py
  01_autotuned_matmul_triton.py
  02_batched_matmul_reference.py
  03_batched_matmul_triton.py
  04_precision_and_accumulation.py
  05_matmul_benchmark_harness.py
```

## Why these stages matter

Matmul is the center of accelerator programming. These files introduce:

- M/N/K shape conventions
- row-major and transposed/strided layouts
- naive CPU reference matmul
- tiled/block CPU reference matmul
- one-program-per-output-tile Triton matmul
- BLOCK_M/BLOCK_N/BLOCK_K
- masks on M/N/K boundaries
- `tl.dot`
- grouped program ordering for L2 reuse
- `@triton.autotune`
- batched matmul
- fp32 accumulation
- benchmark structure

## Design rule

Each important kernel file has:

```text
1. Large comments and interview notes
2. NumPy reference
3. CPU tiled/program-style simulation
4. Real guarded @triton.jit kernel
5. GPU wrapper
6. CPU smoke test
7. Optional GPU correctness helper where relevant
8. Benchmark stub or model
```

## Run CPU-safe tests

```bash
pip install -r requirements.txt
python run_all_smoke_tests.py
```

## Optional GPU/Triton execution

```bash
pip install torch triton
python -m stage06_matmul.03_naive_matmul_triton
python -m stage06_matmul.04_tiled_matmul_triton
python -m stage06_matmul.05_grouped_matmul_triton
python -m stage07_autotuning_batched.01_autotuned_matmul_triton
python -m stage07_autotuning_batched.03_batched_matmul_triton
```

## Integration advice

Copy these folders into your main Triton repo:

```text
stage06_matmul/
stage07_autotuning_batched/
```

Do not delete older files immediately. Move older placeholder-only files into `legacy_notes/` after CPU and GPU tests pass.
