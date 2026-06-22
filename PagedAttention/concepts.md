# Concepts Behind This Repo

## 1. Why Q/K/V starts at `[B, S, D]`
The linear projection changes the **content** of each hidden vector but not the visible rank or visible shape.
So after:

```python
Q_raw = X @ W_q
```

we still have `[B, S, D]`.

Heads are introduced only after we **reshape** the last dimension.

## 2. When heads appear
The head split is:

- before: `[B, S, D]`
- after reshape: `[B, S, H, Dk]`
- after transpose: `[B, H, S, Dk]`

with `D = H * Dk`.

## 3. Why transpose to `[B, H, S, Dk]`
Attention runs independently per head.
Putting `H` before `S` makes the tensor layout easier to reason about for head-wise attention kernels.

## 4. Why prefill must write cache
In real decoding, prompt tokens are not thrown away.
Their K/V must be stored so later decode tokens can attend to them.

## 5. Why paged attention exists
A giant contiguous cache can be hard to grow efficiently.
Paged attention stores K/V in fixed-size blocks/pages and uses a mapping table to find them.
