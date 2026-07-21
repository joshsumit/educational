import torch


# -----------------------------
# Tier 5: Numerical algorithms
# -----------------------------

def online_mean(values: torch.Tensor) -> float:
    # Online mean update in one pass.
    mean = 0.0
    n = 0
    for v in values.reshape(-1):
        n += 1
        mean += (float(v) - mean) / n
    return mean


def online_variance(values: torch.Tensor) -> float:
    # Online variance from running mean and second moment accumulator.
    mean = 0.0
    m2 = 0.0
    n = 0
    for v in values.reshape(-1):
        n += 1
        x = float(v)
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        m2 += delta * delta2
    return m2 / max(n - 1, 1)


def welford_mean(values: torch.Tensor) -> float:
    # Welford mean is same as online mean update.
    return online_mean(values)


def welford_variance(values: torch.Tensor) -> float:
    # Welford variance is numerically stable and one pass.
    return online_variance(values)


def pairwise_reduction_sum(values: torch.Tensor) -> float:
    # Pairwise tree reduction reference.
    arr = [float(v) for v in values.reshape(-1)]
    while len(arr) > 1:
        nxt = []
        for i in range(0, len(arr), 2):
            if i + 1 < len(arr):
                nxt.append(arr[i] + arr[i + 1])
            else:
                nxt.append(arr[i])
        arr = nxt
    return arr[0] if arr else 0.0


def warp_style_reduction(values: torch.Tensor, warp_size: int = 32) -> torch.Tensor:
    # Warp-style reduction simulation: reduce each warp-sized chunk independently.
    flat = values.reshape(-1)
    out = []
    for i in range(0, flat.numel(), warp_size):
        out.append(torch.sum(flat[i : i + warp_size]))
    return torch.stack(out) if out else torch.tensor([], dtype=values.dtype)


def block_reduction(values: torch.Tensor, block_size: int = 256) -> torch.Tensor:
    # Block reduction simulation: reduce each block-sized chunk independently.
    flat = values.reshape(-1)
    out = []
    for i in range(0, flat.numel(), block_size):
        out.append(torch.sum(flat[i : i + block_size]))
    return torch.stack(out) if out else torch.tensor([], dtype=values.dtype)


def segmented_reduction(values: torch.Tensor, segment_ids: torch.Tensor, reduce: str = "sum") -> torch.Tensor:
    # Segmented reduction for values and matching segment id per element.
    num_segments = int(torch.max(segment_ids).item()) + 1 if segment_ids.numel() > 0 else 0
    out = torch.zeros((num_segments,), dtype=values.dtype, device=values.device)
    counts = torch.zeros((num_segments,), dtype=torch.int64, device=values.device)

    if reduce == "max":
        out.fill_(torch.finfo(values.dtype).min)
    if reduce == "min":
        out.fill_(torch.finfo(values.dtype).max)

    for i in range(values.numel()):
        seg = int(segment_ids.reshape(-1)[i])
        val = values.reshape(-1)[i]
        if reduce in ("sum", "mean"):
            out[seg] += val
        elif reduce == "max":
            out[seg] = torch.maximum(out[seg], val)
        elif reduce == "min":
            out[seg] = torch.minimum(out[seg], val)
        counts[seg] += 1

    if reduce == "mean":
        mask = counts > 0
        out[mask] = out[mask] / counts[mask].to(out.dtype)
    return out


def min_reduction(values: torch.Tensor) -> torch.Tensor:
    # Min reduction.
    return torch.min(values)


def max_reduction(values: torch.Tensor) -> torch.Tensor:
    # Max reduction.
    return torch.max(values)


def hillis_steele_scan(values: torch.Tensor) -> torch.Tensor:
    # Inclusive Hillis-Steele scan.
    out = values.clone().to(torch.float32)
    offset = 1
    while offset < out.numel():
        prev = out.clone()
        out[offset:] = prev[offset:] + prev[:-offset]
        offset *= 2
    return out.to(values.dtype)


def inclusive_scan(values: torch.Tensor) -> torch.Tensor:
    # Inclusive scan via cumsum baseline.
    return torch.cumsum(values, dim=0)


def segmented_scan(values: torch.Tensor, segment_starts: torch.Tensor) -> torch.Tensor:
    # Segmented inclusive scan; segment_starts[i]=1 marks a new segment at i.
    out = torch.zeros_like(values)
    running = 0.0
    for i in range(values.numel()):
        if int(segment_starts[i]) == 1:
            running = 0.0
        running += float(values[i])
        out[i] = running
    return out


def branch_divergence_penalty(condition_bits: torch.Tensor, warp_size: int = 32) -> dict:
    # Branch divergence simulation for one warp.
    # If threads within a warp follow different branches, hardware serializes paths.
    if condition_bits.numel() == 0:
        return {"active_threads": 0, "divergent_threads": 0, "estimated_penalty": 1.0}

    warp = condition_bits.reshape(-1)[:warp_size].to(torch.bool)
    true_count = int(torch.sum(warp).item())
    false_count = int(warp.numel() - true_count)
    divergent = min(true_count, false_count)

    # Rough model: fully uniform branch is ~1x, split branches approach ~2x.
    penalty = 1.0 if divergent == 0 else 1.0 + (divergent / max(warp.numel(), 1))
    return {
        "active_threads": int(warp.numel()),
        "divergent_threads": divergent,
        "estimated_penalty": float(penalty),
    }


# -----------------------------
# Tier 6: GPU kernel building blocks
# -----------------------------

def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Elementwise vector add.
    return a + b


def vector_scale(a: torch.Tensor, alpha: float) -> torch.Tensor:
    # Scale vector by scalar.
    return a * alpha


def saxpy(alpha: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # SAXPY: y = alpha * x + y.
    return alpha * x + y


def matrix_copy(a: torch.Tensor) -> torch.Tensor:
    # Matrix copy reference.
    return a.clone()


def matrix_transpose(a: torch.Tensor) -> torch.Tensor:
    # Matrix transpose baseline.
    return a.transpose(0, 1)


def tiled_matrix_transpose(a: torch.Tensor, tile: int = 32) -> torch.Tensor:
    # Tiled transpose simulation.
    h, w = a.shape
    out = torch.empty((w, h), dtype=a.dtype, device=a.device)
    for i in range(0, h, tile):
        for j in range(0, w, tile):
            block = a[i : i + tile, j : j + tile]
            out[j : j + block.shape[1], i : i + block.shape[0]] = block.transpose(0, 1)
    return out


def histogram_naive(values: torch.Tensor, num_bins: int, min_val: float, max_val: float) -> torch.Tensor:
    # Naive histogram with scalar updates.
    hist = torch.zeros((num_bins,), dtype=torch.int64)
    scale = num_bins / max(max_val - min_val, 1e-12)
    for v in values.reshape(-1):
        b = int((float(v) - min_val) * scale)
        b = max(0, min(num_bins - 1, b))
        hist[b] += 1
    return hist


def parallel_histogram_simulation(values: torch.Tensor, num_bins: int, min_val: float, max_val: float, chunks: int = 8) -> torch.Tensor:
    # Parallel histogram simulation: local chunk histograms + final reduction.
    local_hists = []
    flat = values.reshape(-1)
    chunk_size = (flat.numel() + chunks - 1) // chunks
    for i in range(0, flat.numel(), chunk_size):
        local_hists.append(histogram_naive(flat[i : i + chunk_size], num_bins, min_val, max_val))
    return torch.stack(local_hists, dim=0).sum(dim=0)


def prefix_scan_kernel(values: torch.Tensor) -> torch.Tensor:
    # Prefix scan kernel reference.
    return inclusive_scan(values)


def reduction_kernel(values: torch.Tensor) -> torch.Tensor:
    # Reduction kernel reference.
    return torch.sum(values)


def gather_kernel(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    # Gather kernel reference.
    return values[indices]


def scatter_kernel(base: torch.Tensor, indices: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    # Scatter kernel reference.
    out = base.clone()
    out[indices] = src
    return out


def shared_memory_tiling_example(a: torch.Tensor, b: torch.Tensor, tile: int = 16) -> torch.Tensor:
    # Shared-memory tiling example via tiled matmul scheduling.
    # This simulates tile loads and tile compute in Python.
    m, k_a = a.shape
    k_b, n = b.shape
    if k_a != k_b:
        raise ValueError("Inner dimensions mismatch")

    out = torch.zeros((m, n), dtype=a.dtype, device=a.device)
    for i0 in range(0, m, tile):
        for j0 in range(0, n, tile):
            for k0 in range(0, k_a, tile):
                a_tile = a[i0 : i0 + tile, k0 : k0 + tile]
                b_tile = b[k0 : k0 + tile, j0 : j0 + tile]
                out[i0 : i0 + a_tile.shape[0], j0 : j0 + b_tile.shape[1]] += a_tile @ b_tile
    return out


def bank_conflict_simulation(indices: torch.Tensor, num_banks: int = 32) -> torch.Tensor:
    # Bank conflict simulation: count accesses per bank per "warp" row.
    # indices shape [W, L] where each row is one warp-like access set.
    banks = indices % num_banks
    conflicts = torch.zeros((indices.shape[0],), dtype=torch.int64)
    for w in range(indices.shape[0]):
        unique, counts = torch.unique(banks[w], return_counts=True)
        conflicts[w] = torch.sum(torch.clamp(counts - 1, min=0))
    return conflicts


def coalesced_access_example(base: torch.Tensor, stride: int) -> torch.Tensor:
    # Coalesced access example: gather strided addresses to illustrate memory pattern.
    idx = torch.arange(0, base.numel(), stride, dtype=torch.int64, device=base.device)
    return base[idx]


def memory_hierarchy_analysis(total_flops: float, total_bytes: float, gpu_peak_tflops: float, dram_bandwidth_gbps: float) -> dict:
    # Roofline-style analysis helper.
    # Arithmetic intensity (FLOPs/byte) predicts whether kernel is memory or compute bound.
    if total_bytes <= 0:
        raise ValueError("total_bytes must be positive")
    if gpu_peak_tflops <= 0 or dram_bandwidth_gbps <= 0:
        raise ValueError("Peak compute and bandwidth must be positive")

    intensity = total_flops / total_bytes
    peak_flops = gpu_peak_tflops * 1e12
    peak_bw = dram_bandwidth_gbps * 1e9
    bw_limited_flops = intensity * peak_bw
    bottleneck = "compute" if bw_limited_flops >= peak_flops else "memory"

    return {
        "arithmetic_intensity": float(intensity),
        "compute_peak_flops": float(peak_flops),
        "dram_peak_bytes_per_sec": float(peak_bw),
        "bandwidth_limited_flops": float(bw_limited_flops),
        "predicted_bottleneck": bottleneck,
    }


def roofline_analysis_matmul(m: int, n: int, k: int, dtype_bytes: int = 4, gpu_peak_tflops: float = 60.0, dram_bandwidth_gbps: float = 1000.0) -> dict:
    # Convenience roofline analysis for GEMM-like workload C[M,N] = A[M,K] @ B[K,N].
    if min(m, n, k, dtype_bytes) <= 0:
        raise ValueError("m, n, k and dtype_bytes must be positive")

    # GEMM FLOPs: 2 * M * N * K for multiply-add.
    total_flops = float(2 * m * n * k)
    # Approximate bytes moved: read A and B once, read+write C once.
    total_bytes = float((m * k + k * n + 2 * m * n) * dtype_bytes)

    out = memory_hierarchy_analysis(
        total_flops=total_flops,
        total_bytes=total_bytes,
        gpu_peak_tflops=gpu_peak_tflops,
        dram_bandwidth_gbps=dram_bandwidth_gbps,
    )
    out["m"] = m
    out["n"] = n
    out["k"] = k
    return out


def compute_occupancy_estimate(
    threads_per_block: int,
    registers_per_thread: int,
    shared_mem_bytes_per_block: int,
    max_threads_per_sm: int = 2048,
    max_registers_per_sm: int = 65536,
    max_shared_mem_per_sm: int = 98304,
    max_blocks_per_sm: int = 32,
) -> dict:
    # Occupancy estimate from thread/register/shared-memory constraints.
    # This is a platform-agnostic model and not a replacement for vendor profilers.
    if threads_per_block <= 0:
        raise ValueError("threads_per_block must be positive")
    if registers_per_thread <= 0:
        raise ValueError("registers_per_thread must be positive")
    if shared_mem_bytes_per_block < 0:
        raise ValueError("shared_mem_bytes_per_block must be non-negative")

    by_threads = max_threads_per_sm // threads_per_block
    by_registers = max_registers_per_sm // (registers_per_thread * threads_per_block)
    if shared_mem_bytes_per_block == 0:
        by_shared = max_blocks_per_sm
    else:
        by_shared = max_shared_mem_per_sm // shared_mem_bytes_per_block

    resident_blocks = max(0, min(by_threads, by_registers, by_shared, max_blocks_per_sm))
    resident_threads = resident_blocks * threads_per_block
    occupancy = resident_threads / max_threads_per_sm

    return {
        "resident_blocks_per_sm": int(resident_blocks),
        "resident_threads_per_sm": int(resident_threads),
        "occupancy": float(occupancy),
        "limit_by_threads": int(by_threads),
        "limit_by_registers": int(by_registers),
        "limit_by_shared_memory": int(by_shared),
    }


def atomic_add_simulation(base: torch.Tensor, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    # Atomic-add style accumulation reference.
    # In real GPU code, each update would be protected with atomicAdd.
    if indices.numel() != values.numel():
        raise ValueError("indices and values must have the same number of elements")

    out = base.clone()
    for idx, val in zip(indices.reshape(-1), values.reshape(-1)):
        out[int(idx)] += val
    return out


def barrier_synchronization_simulation(stage_outputs: list[torch.Tensor]) -> torch.Tensor:
    # Barrier concept simulation: all thread groups finish stage N before stage N+1 combine.
    # Here we simply aggregate after all partial tensors are available.
    if len(stage_outputs) == 0:
        raise ValueError("stage_outputs must not be empty")
    acc = torch.zeros_like(stage_outputs[0])
    for part in stage_outputs:
        acc = acc + part
    return acc


def double_buffering_simulation(chunks_a: list[torch.Tensor], chunks_b: list[torch.Tensor]) -> list[torch.Tensor]:
    # Double buffering simulation with ping-pong buffer IDs.
    if len(chunks_a) != len(chunks_b):
        raise ValueError("chunk lists must have the same length")

    outputs = []
    ping = 0
    for i in range(len(chunks_a)):
        # "Load" next chunk in one buffer while "compute" previous in the other.
        _load_buffer = ping
        _compute_buffer = 1 - ping
        outputs.append(chunks_a[i] @ chunks_b[i])
        ping = 1 - ping
    return outputs


# -----------------------------
# Tier 7: Advanced GEMM
# -----------------------------

def blocked_gemm(a: torch.Tensor, b: torch.Tensor, block: int = 64) -> torch.Tensor:
    # Blocked GEMM reference.
    return shared_memory_tiling_example(a, b, tile=block)


def register_tiled_gemm(a: torch.Tensor, b: torch.Tensor, mr: int = 4, nr: int = 4) -> torch.Tensor:
    # Register-tiled GEMM simulation using micro-tiles (mr x nr).
    m, k_a = a.shape
    k_b, n = b.shape
    if k_a != k_b:
        raise ValueError("Inner dimensions mismatch")

    out = torch.zeros((m, n), dtype=a.dtype, device=a.device)
    for i0 in range(0, m, mr):
        for j0 in range(0, n, nr):
            acc = torch.zeros((min(mr, m - i0), min(nr, n - j0)), dtype=a.dtype, device=a.device)
            for k in range(k_a):
                a_vec = a[i0 : i0 + acc.shape[0], k].unsqueeze(1)
                b_vec = b[k, j0 : j0 + acc.shape[1]].unsqueeze(0)
                acc += a_vec * b_vec
            out[i0 : i0 + acc.shape[0], j0 : j0 + acc.shape[1]] = acc
    return out


def packed_gemm(a: torch.Tensor, b: torch.Tensor, pack_k: int = 32) -> torch.Tensor:
    # Packed GEMM simulation; pack B in K-chunks for improved locality.
    m, k_a = a.shape
    k_b, n = b.shape
    if k_a != k_b:
        raise ValueError("Inner dimensions mismatch")

    out = torch.zeros((m, n), dtype=a.dtype, device=a.device)
    for k0 in range(0, k_a, pack_k):
        k1 = min(k0 + pack_k, k_a)
        b_pack = b[k0:k1].contiguous()
        out += a[:, k0:k1] @ b_pack
    return out


def batched_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Batched GEMM for [B, M, K] x [B, K, N].
    return torch.bmm(a, b)


def strided_batched_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Strided batched GEMM reference (same as batched in dense tensor form).
    return torch.bmm(a, b)


def sparse_gemm(a_sparse: torch.Tensor, b_dense: torch.Tensor) -> torch.Tensor:
    # Sparse GEMM with sparse-dense matmul.
    return torch.sparse.mm(a_sparse, b_dense)


def tensor_core_style_gemm_simulation(a: torch.Tensor, b: torch.Tensor, mma_tile: int = 16) -> torch.Tensor:
    # Tensor Core style simulation by MMA-like tile accumulation.
    return blocked_gemm(a, b, block=mma_tile)


def systolic_array_gemm_simulation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Systolic array simulation reference using standard matmul correctness path.
    return a @ b


def output_stationary_dataflow(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Output-stationary dataflow simulation.
    return a @ b


def weight_stationary_dataflow(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Weight-stationary dataflow simulation.
    return a @ b


def row_stationary_dataflow(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Row-stationary dataflow simulation.
    return a @ b


# -----------------------------
# Practice Questions
# -----------------------------
# Beginner:
# 1) Why is Welford variance more stable than naive variance formulas?
# 2) What are inclusive vs segmented scans used for in ML workloads?
# 3) Why is tiled matrix transpose preferred over naive transpose on GPU?
# Intermediate:
# 4) Compare histogram_naive and parallel_histogram_simulation bottlenecks.
# 5) How do bank conflicts arise and how does padding help?
# 6) Why does blocked GEMM generally improve cache reuse?
# Advanced:
# 7) Compare blocked_gemm, register_tiled_gemm, and packed_gemm by data movement.
# 8) How would you choose tile sizes for tensor_core_style_gemm_simulation?
# 9) What are differences between output-stationary and weight-stationary dataflows?
# Expert:
# 10) Design a roofline-based experiment for GEMM variants in this file.
# 11) Explain double buffering benefits and hazards in real kernels.
# 12) How would you implement deterministic multi-block reduction with minimal overhead?
