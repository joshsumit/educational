# Stage 2 and 3 Integration Guide

## Should I delete original files?

No.

Your original files are useful as earlier learning notes. Treat this package as a cleaner second wave.

Recommended integration:

```text
current repo/
  triton/
    stage00_environment/
    stage01_programming_model/
    stage02_memory_and_masks/      <- copy/replace from this zip
    stage03_elementwise/           <- copy/replace from this zip
```

## Keep NumPy or remove it?

Keep NumPy.

Each serious Triton file should keep a NumPy reference because it gives you:

1. a correctness oracle
2. CPU-only tests
3. interview-friendly explanation
4. easier debugging when Triton output differs
5. a baseline for benchmarking

The right final pattern is not "Triton only". The right pattern is:

```text
reference implementation
blocked CPU simulation
real Triton kernel
wrapper
correctness test
benchmark stub
performance notes
```

## What can be deprecated later?

Later, once the new files pass tests, older placeholder-only files can be moved to:

```text
legacy_notes/
```

Do not delete them immediately. Move them only after your new smoke tests and GPU tests pass.
