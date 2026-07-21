# Stage 1 Interview Questions

## 1. What is `tl.program_id()`?

`tl.program_id(axis)` returns the coordinate of the current Triton program along a launch-grid axis.
You use it to decide which block or tile of data the current program owns.

Good answer:

```text
A Triton kernel launches a grid of programs. `tl.program_id(0)` gives the current program's coordinate
along axis 0. In a vector kernel, I use it to compute offsets like `pid * BLOCK + tl.arange(0, BLOCK)`.
```

## 2. Is a Triton program the same as a CUDA thread?

No.

CUDA exposes individual threads directly. Triton encourages a block/program-level view where one program
works on a vector or tile of elements.

Good answer:

```text
CUDA is thread-centric. Triton is program/tile-centric. I decide what data tile a program owns, then express
vectorized loads, computation, reductions, and stores over that tile.
```

## 3. Why do we need masks?

Masks protect boundary accesses.

If `N=10` and `BLOCK=8`, program 1 creates offsets:

```text
[8, 9, 10, 11, 12, 13, 14, 15]
```

Only offsets 8 and 9 are valid. The mask is:

```text
[T, T, F, F, F, F, F, F]
```

## 4. Why use `triton.cdiv(n, BLOCK)`?

Because the last block may be partial. `cdiv` ensures the grid launches enough programs to cover all
elements.

Example:

```text
n = 1000
BLOCK = 256
cdiv(1000, 256) = 4
```

## 5. What is the difference between 1D, 2D, and 3D grids?

- 1D grids are natural for vectors and flattened tensors.
- 2D grids are natural for matrix tiles and matmul outputs.
- 3D grids are useful for batched and multi-head workloads.

## 6. Why does grouped program ordering matter?

Grouped ordering can improve cache locality, especially for matmul. By processing several neighboring
M tiles for the same N tile, adjacent programs can reuse B tiles from cache more effectively.

## 7. How should you explain a simple Triton kernel in an interview?

Use this sequence:

```text
1. Define what one program owns.
2. Compute offsets from program_id and arange.
3. Build masks for boundary safety.
4. Load data using masked loads.
5. Compute in registers/on tile values.
6. Store results using masked stores.
7. Explain memory access pattern and expected bottleneck.
```
