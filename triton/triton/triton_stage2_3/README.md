# Triton From Zero To Expert — Stage 2 and Stage 3

This is the second downloadable implementation wave.

It adds the first real kernel layer after Stage 0/1:

- **Stage 2 — Memory, masks, pointer arithmetic, strides, and coalescing**
- **Stage 3 — Elementwise and fused elementwise Triton kernels**

Design rule used in every kernel file:

```text
CPU NumPy reference
+ CPU blocked simulation
+ real @triton.jit kernel, guarded by optional imports
+ wrapper function
+ correctness helper
+ benchmark stub
+ extensive comments
```

## Should this replace your original files?

No. Do not delete your original repo yet.

Use this as an additive update:

1. Keep your current `triton/` folder as backup.
2. Copy `stage02_memory_and_masks/` and `stage03_elementwise/` from this repo into your working repo.
3. Update your top-level `run_all_smoke_tests.py` to include the new modules.
4. Once smoke tests pass, you can deprecate older placeholder files gradually.

The NumPy versions are intentionally included. They are useful for:

- CPU-only smoke tests
- parity checks
- interview explanation
- debugging Triton kernels
- showing the exact mathematical reference before optimization

## Run CPU-safe tests

```bash
pip install -r requirements.txt
python run_all_smoke_tests.py
```

## Optional GPU/Triton execution

```bash
pip install torch triton
python -m stage03_elementwise.00_vector_add_triton
python -m stage03_elementwise.01_relu_triton
python -m stage03_elementwise.02_silu_triton
python -m stage03_elementwise.03_gelu_triton
python -m stage03_elementwise.04_scale_bias_triton
python -m stage03_elementwise.05_fused_affine_relu_triton
```
