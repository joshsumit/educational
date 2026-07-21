# Triton From Zero To Expert — Stage 0 and Stage 1

This repository is the first downloadable implementation wave of the Triton learning path.

It covers:

- Stage 0: environment, mental model, CUDA-vs-Triton comparison, first-kernel anatomy, common primitives.
- Stage 1: program ids, block vectors, masks, launch grids, 2D grids, 3D grids, and grouped program ordering.

The design goal is interview preparation for low-level AI systems roles at NVIDIA, AMD, Qualcomm,
Cerebras, Tenstorrent, compiler/runtime teams, and inference-runtime teams.

## Why Stage 0 and 1 are intentionally Python-first

Stage 0 and Stage 1 teach the execution model before adding heavy kernels. Every Python simulation mirrors
the mental model used in real Triton kernels:

```text
program_id -> block offsets -> masks -> load/store ownership -> grid mapping
```

You should run these on any CPU-only machine first. If `torch` and `triton` are installed, the environment
report will detect them, but the smoke tests do not require a GPU.

## Install

```bash
pip install -r requirements.txt
python run_all_smoke_tests.py
```

Optional GPU/Triton environment:

```bash
pip install torch triton
python -m stage00_environment.03_optional_imports
```

## Directory layout

```text
stage00_environment/
  00_what_is_triton.md
  01_gpu_execution_model.py
  02_triton_vs_cuda.py
  03_optional_imports.py
  04_first_kernel_anatomy.py
  05_common_triton_primitives.py

stage01_programming_model/
  00_program_ids.py
  01_block_vectors.py
  02_masks.py
  03_grid_mapping.py
  04_2d_grids.py
  05_3d_grids.py
  06_program_ordering.py
  07_interview_questions.md
```

## Study order

1. Read `stage00_environment/00_what_is_triton.md`.
2. Run `python run_all_smoke_tests.py`.
3. Read the comments inside each Python file.
4. Modify block sizes and tensor sizes.
5. Explain every offset and every mask without looking at the code.

## What comes next

The next downloadable wave should implement Stage 2 and Stage 3:

- real `@triton.jit` vector add
- masked load/store kernels
- ReLU, SiLU, GELU
- scale+bias fusion
- correctness tests
- benchmark stubs
