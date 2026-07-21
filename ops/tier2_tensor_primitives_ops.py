import torch


# -----------------------------
# Indexing operations
# -----------------------------

def gather_naive_2d(x: torch.Tensor, index: torch.Tensor, dim: int) -> torch.Tensor:
    # Naive 2D gather for learning index semantics.
    if x.dim() != 2 or index.dim() != 2:
        raise ValueError("gather_naive_2d expects 2D tensors")

    out = torch.empty_like(index, dtype=x.dtype)
    if dim == 0:
        for i in range(index.shape[0]):
            for j in range(index.shape[1]):
                out[i, j] = x[int(index[i, j]), j]
    elif dim == 1:
        for i in range(index.shape[0]):
            for j in range(index.shape[1]):
                out[i, j] = x[i, int(index[i, j])]
    else:
        raise ValueError("dim must be 0 or 1")
    return out


def gather_optimized(x: torch.Tensor, index: torch.Tensor, dim: int) -> torch.Tensor:
    # Optimized gather using backend gather kernel.
    return torch.gather(x, dim=dim, index=index)


def scatter_naive_2d(base: torch.Tensor, index: torch.Tensor, src: torch.Tensor, dim: int) -> torch.Tensor:
    # Naive scatter write; if repeated indices exist, last write wins.
    out = base.clone()
    if dim == 0:
        for i in range(index.shape[0]):
            for j in range(index.shape[1]):
                out[int(index[i, j]), j] = src[i, j]
    elif dim == 1:
        for i in range(index.shape[0]):
            for j in range(index.shape[1]):
                out[i, int(index[i, j])] = src[i, j]
    else:
        raise ValueError("dim must be 0 or 1")
    return out


def scatter_optimized(base: torch.Tensor, index: torch.Tensor, src: torch.Tensor, dim: int) -> torch.Tensor:
    # Optimized scatter path.
    out = base.clone()
    return out.scatter_(dim=dim, index=index, src=src)


def scatter_add_naive_2d(base: torch.Tensor, index: torch.Tensor, src: torch.Tensor, dim: int) -> torch.Tensor:
    # Naive scatter-add accumulation.
    out = base.clone()
    if dim == 0:
        for i in range(index.shape[0]):
            for j in range(index.shape[1]):
                out[int(index[i, j]), j] += src[i, j]
    elif dim == 1:
        for i in range(index.shape[0]):
            for j in range(index.shape[1]):
                out[i, int(index[i, j])] += src[i, j]
    else:
        raise ValueError("dim must be 0 or 1")
    return out


def scatter_add_optimized(base: torch.Tensor, index: torch.Tensor, src: torch.Tensor, dim: int) -> torch.Tensor:
    # Optimized scatter-add path.
    out = base.clone()
    return out.scatter_add_(dim=dim, index=index, src=src)


def scatter_add_race_conditions(index: torch.Tensor) -> dict:
    # Analyze write-conflict risk for scatter/scatter_add style kernels.
    # Duplicate indices imply concurrent writers and require atomic behavior.
    flat = index.reshape(-1).to(torch.int64)
    unique, counts = torch.unique(flat, return_counts=True)
    max_conflict = int(torch.max(counts).item()) if counts.numel() > 0 else 0
    conflict_writes = int(torch.sum(torch.clamp(counts - 1, min=0)).item()) if counts.numel() > 0 else 0
    return {
        "num_indices": int(flat.numel()),
        "num_unique": int(unique.numel()),
        "max_writers_to_one_index": max_conflict,
        "extra_conflicting_writes": conflict_writes,
        "needs_atomic": bool(max_conflict > 1),
    }


def coalescing_analysis(index_map: torch.Tensor, warp_size: int = 32) -> dict:
    # Analyze whether first warp accesses contiguous addresses.
    # This is a simple proxy for coalesced global-memory transactions.
    flat = index_map.reshape(-1).to(torch.int64)
    if flat.numel() == 0:
        return {"is_coalesced": True, "mean_stride": 0.0, "max_stride": 0}

    warp = flat[:warp_size]
    if warp.numel() <= 1:
        return {"is_coalesced": True, "mean_stride": 0.0, "max_stride": 0}

    strides = torch.abs(warp[1:] - warp[:-1])
    mean_stride = float(torch.mean(strides.to(torch.float32)).item())
    max_stride = int(torch.max(strides).item())
    is_coalesced = bool(torch.all(strides == 1))
    return {
        "is_coalesced": is_coalesced,
        "mean_stride": mean_stride,
        "max_stride": max_stride,
    }


def segment_reduction_naive(values: torch.Tensor, segment_ids: torch.Tensor, num_segments: int, reduce: str = "sum") -> torch.Tensor:
    # Naive segment reduction where each value belongs to one segment.
    out = torch.zeros((num_segments,), dtype=values.dtype, device=values.device)
    if reduce == "max":
        out.fill_(torch.finfo(values.dtype).min)
    if reduce == "min":
        out.fill_(torch.finfo(values.dtype).max)

    counts = torch.zeros((num_segments,), dtype=torch.int64, device=values.device)
    for i in range(values.numel()):
        seg = int(segment_ids[i])
        val = values.reshape(-1)[i]
        if reduce in ("sum", "mean"):
            out[seg] += val
        elif reduce == "max":
            out[seg] = torch.maximum(out[seg], val)
        elif reduce == "min":
            out[seg] = torch.minimum(out[seg], val)
        else:
            raise ValueError("reduce must be sum|mean|max|min")
        counts[seg] += 1

    if reduce == "mean":
        mask = counts > 0
        out[mask] = out[mask] / counts[mask].to(out.dtype)
    return out


def segment_reduction_optimized(values: torch.Tensor, segment_ids: torch.Tensor, num_segments: int, reduce: str = "sum") -> torch.Tensor:
    # Optimized-ish reference using scatter_add for sum/mean.
    if reduce in ("sum", "mean"):
        out = torch.zeros((num_segments,), dtype=values.dtype, device=values.device)
        out.scatter_add_(0, segment_ids.reshape(-1), values.reshape(-1))
        if reduce == "mean":
            counts = torch.zeros((num_segments,), dtype=values.dtype, device=values.device)
            ones = torch.ones_like(values, dtype=values.dtype)
            counts.scatter_add_(0, segment_ids.reshape(-1), ones.reshape(-1))
            mask = counts > 0
            out[mask] = out[mask] / counts[mask]
        return out
    return segment_reduction_naive(values, segment_ids, num_segments, reduce=reduce)


def argmax_naive(x: torch.Tensor, dim: int) -> torch.Tensor:
    # Naive argmax by reduction along dim.
    return torch.argmax(x, dim=dim)


def argmax_optimized(x: torch.Tensor, dim: int) -> torch.Tensor:
    # Optimized argmax backend path.
    return torch.argmax(x, dim=dim)


def argmin_naive(x: torch.Tensor, dim: int) -> torch.Tensor:
    # Naive argmin by reduction along dim.
    return torch.argmin(x, dim=dim)


def argmin_optimized(x: torch.Tensor, dim: int) -> torch.Tensor:
    # Optimized argmin backend path.
    return torch.argmin(x, dim=dim)


def topk_naive_1d(x: torch.Tensor, k: int, largest: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    # Naive top-k for 1D tensor via Python-side sort.
    pairs = [(float(x[i]), i) for i in range(x.numel())]
    pairs.sort(key=lambda p: p[0], reverse=largest)
    selected = pairs[:k]
    values = torch.tensor([p[0] for p in selected], dtype=x.dtype, device=x.device)
    indices = torch.tensor([p[1] for p in selected], dtype=torch.int64, device=x.device)
    return values, indices


def topk_optimized(x: torch.Tensor, k: int, dim: int = -1, largest: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    # Optimized top-k backend path.
    return torch.topk(x, k, dim=dim, largest=largest)


def sort_naive_1d(x: torch.Tensor, descending: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    # Naive sort for 1D tensor.
    pairs = [(float(x[i]), i) for i in range(x.numel())]
    pairs.sort(key=lambda p: p[0], reverse=descending)
    values = torch.tensor([p[0] for p in pairs], dtype=x.dtype, device=x.device)
    indices = torch.tensor([p[1] for p in pairs], dtype=torch.int64, device=x.device)
    return values, indices


def sort_optimized(x: torch.Tensor, dim: int = -1, descending: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    # Optimized sort backend path.
    return torch.sort(x, dim=dim, descending=descending)


# -----------------------------
# Shape operations
# -----------------------------

def reshape_op(x: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    # Reshape may return a view or copy depending on memory layout.
    return torch.reshape(x, shape)


def view_op(x: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    # View requires compatible contiguous-like memory strides.
    return x.view(*shape)


def flatten_op(x: torch.Tensor, start_dim: int = 0, end_dim: int = -1) -> torch.Tensor:
    # Flatten range of dims into one dim.
    return torch.flatten(x, start_dim=start_dim, end_dim=end_dim)


def squeeze_op(x: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    # Remove singleton dimensions.
    return x.squeeze() if dim is None else x.squeeze(dim)


def unsqueeze_op(x: torch.Tensor, dim: int) -> torch.Tensor:
    # Insert singleton dimension.
    return x.unsqueeze(dim)


def transpose_op(x: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    # Swap two dimensions.
    return x.transpose(dim0, dim1)


def permute_op(x: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    # Arbitrary dimension permutation.
    return x.permute(*dims)


def slice_op(x: torch.Tensor, starts: tuple[int, ...], ends: tuple[int, ...]) -> torch.Tensor:
    # Generic slice helper for up to 4D tensors.
    if x.dim() != len(starts) or x.dim() != len(ends):
        raise ValueError("starts/ends must match tensor rank")
    slices = tuple(slice(starts[i], ends[i]) for i in range(x.dim()))
    return x[slices]


def concatenate_op(tensors: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
    # Concatenate a list of tensors along one dimension.
    return torch.cat(tensors, dim=dim)


def split_op(x: torch.Tensor, split_size_or_sections, dim: int = 0) -> tuple[torch.Tensor, ...]:
    # Split tensor into chunks by size or section list.
    return torch.split(x, split_size_or_sections, dim=dim)


def chunk_op(x: torch.Tensor, chunks: int, dim: int = 0) -> tuple[torch.Tensor, ...]:
    # Chunk tensor into equal count parts (last may be smaller).
    return torch.chunk(x, chunks, dim=dim)


def broadcast_to_op(x: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    # Broadcast tensor to target shape.
    return torch.broadcast_to(x, shape)


def expand_op(x: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    # Expand singleton dimensions without data copy.
    return x.expand(*shape)


# -----------------------------
# Practice Questions
# -----------------------------
# Beginner:
# 1) What is the difference between gather and scatter?
# 2) Why is scatter_add needed when duplicate indices exist?
# 3) What is the practical difference between reshape and view?
# Intermediate:
# 4) How would you implement top-k in O(n log k) rather than full sort?
# 5) Why can expand be zero-copy while broadcast_to may materialize in downstream ops?
# 6) How do split and chunk differ for uneven lengths?
# Advanced:
# 7) How would you optimize segmented reductions for skewed segment sizes?
# 8) What memory-access patterns make gather/scatter slow on GPU?
# 9) How would you detect and avoid index out-of-bounds in custom kernels efficiently?
# Expert:
# 10) Design a fused gather-transform-scatter kernel for sparse updates.
# 11) Explain deterministic scatter_add across parallel workers.
# 12) How would you benchmark permutation-heavy pipelines for cache locality?
