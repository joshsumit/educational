# 00 — What Is Triton?

Triton is a Python-based language and compiler for writing custom GPU kernels. It is commonly used when
a PyTorch operation is too slow, too memory-heavy, or too generic for a specific model/runtime path.

## The simple mental model

A Triton kernel launches a grid of **programs**.

Each program usually owns a block/tile of data:

```text
Program 0 -> elements 0..BLOCK-1
Program 1 -> elements BLOCK..2*BLOCK-1
Program 2 -> elements 2*BLOCK..3*BLOCK-1
```

In CUDA, beginners often think in terms of individual threads. In Triton, beginners should first think in
terms of block vectors:

```python
pid = tl.program_id(0)
offs = pid * BLOCK + tl.arange(0, BLOCK)
```

That one line says:

```text
I am program `pid`.
I own a vector of BLOCK logical element offsets.
I will load, compute, and store those offsets, usually under a boundary mask.
```

## Why Triton matters for low-level AI interviews

Kernel and runtime interviews frequently focus on:

- pointer arithmetic
- boundary masks
- memory coalescing
- reductions
- normalization kernels
- tiled matmul
- attention kernels
- quantized matmul
- benchmarking and profiling

Triton is an excellent study vehicle because it exposes those concepts without requiring handwritten CUDA
for every example.

## Where Triton fits in modern AI systems

A common production path looks like this:

```text
model code
  -> PyTorch graph
  -> compiler/runtime lowering
  -> generated/fused kernels
  -> Triton/CUDA/HIP kernels
  -> GPU execution
```

For your interview goal, the key is not just knowing syntax. The key is explaining how a high-level tensor
operation becomes blocks of memory movement and arithmetic on accelerator hardware.
