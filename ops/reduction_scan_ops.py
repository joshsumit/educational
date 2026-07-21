import torch


def reduction_sum_naive(row: torch.Tensor) -> float:
    # Sequential sum reduction.
    total = 0.0
    for item in row:
        total += float(item)
    return total


def reduction_sum_tree(row: torch.Tensor) -> float:
    # Tree-style pairwise reduction simulation.
    partial = [float(v) for v in row]
    while len(partial) > 1:
        next_partial = []
        for i in range(0, len(partial), 2):
            if i + 1 < len(partial):
                next_partial.append(partial[i] + partial[i + 1])
            else:
                next_partial.append(partial[i])
        partial = next_partial
    return partial[0]


def reduction_sum_kahan(row: torch.Tensor) -> float:
    # Kahan compensated summation for improved floating-point accuracy.
    total = 0.0
    comp = 0.0
    for item in row:
        value = float(item)
        y = value - comp
        t = total + y
        comp = (t - total) - y
        total = t
    return total


def prefix_sum_naive(row: torch.Tensor) -> torch.Tensor:
    # Inclusive prefix sum: out[i] = sum(row[0:i+1]).
    out = torch.zeros_like(row)
    running = 0.0
    for i, item in enumerate(row):
        running += float(item)
        out[i] = running
    return out


def prefix_sum_blelloch_exclusive(row: torch.Tensor) -> torch.Tensor:
    # Blelloch exclusive scan (power-of-two padded) on 1D tensors.
    n = row.numel()
    if n == 0:
        return row.clone()

    size = 1
    while size < n:
        size <<= 1
    work = [0.0] * size
    for i in range(n):
        work[i] = float(row[i])

    stride = 1
    while stride < size:
        step = stride * 2
        for i in range(step - 1, size, step):
            work[i] += work[i - stride]
        stride = step

    work[size - 1] = 0.0

    stride = size // 2
    while stride >= 1:
        step = stride * 2
        for i in range(step - 1, size, step):
            left = i - stride
            temp = work[left]
            work[left] = work[i]
            work[i] = work[i] + temp
        stride //= 2

    return torch.tensor(work[:n], dtype=row.dtype, device=row.device)


# -----------------------------
# Practice Questions
# -----------------------------
# Beginner:
# 1) What is the difference between inclusive and exclusive scan?
# 2) Why can tree reduction be faster than sequential reduction in parallel hardware?
# 3) What error pattern does Kahan summation reduce?
# Intermediate:
# 4) Why does Blelloch scan use up-sweep and down-sweep phases?
# 5) How would you handle non-power-of-two lengths efficiently in Blelloch scan?
# 6) Compare numerical stability of reduction_sum_naive vs reduction_sum_tree.
# Advanced:
# 7) How would you map prefix_sum_blelloch_exclusive to warps and thread blocks?
# 8) Where do bank conflicts appear in shared-memory scan implementations?
# 9) How do segmented scans relate to ragged batch processing in NLP?
# Expert:
# 10) Design a hierarchical multi-block scan with minimal global synchronization.
# 11) How would you fuse scan with map operations to reduce memory traffic?
# 12) Explain deterministic reduction strategies across distributed devices.

