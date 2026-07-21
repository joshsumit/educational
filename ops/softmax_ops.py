import math
import torch


def softmax_naive_unsafe(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # Unsafe baseline softmax: exp(x) / sum(exp(x)).
    exp_x = torch.exp(x)
    denom = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / denom


def softmax_safe(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # Numerically stable softmax: subtract max before exponentiation.
    row_max = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - row_max)
    denom = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / denom


def softmax_online_rowwise(x: torch.Tensor) -> torch.Tensor:
    # Online row-wise softmax for 2D tensors [M, N].
    out_rows = []
    for row in x:
        running_max = float("-inf")
        running_sum = 0.0

        for item in row:
            value = float(item)
            if value > running_max:
                running_sum = running_sum * math.exp(running_max - value) + 1.0
                running_max = value
            else:
                running_sum += math.exp(value - running_max)

        out_rows.append(torch.exp(row - running_max) / running_sum)
    return torch.stack(out_rows, dim=0)


def masked_softmax_naive(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # Naive masked softmax for boolean mask where True means keep.
    neg_inf = torch.full_like(scores, -1e9)
    masked_scores = torch.where(mask, scores, neg_inf)
    return softmax_safe(masked_scores, dim=dim)


def masked_softmax_online(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Online masked softmax for 2D tensors [M, N].
    out_rows = []
    for row, row_mask in zip(scores, mask):
        running_max = float("-inf")
        running_sum = 0.0

        for item, keep in zip(row, row_mask):
            if not bool(keep):
                continue
            value = float(item)
            if value > running_max:
                running_sum = running_sum * math.exp(running_max - value) + 1.0
                running_max = value
            else:
                running_sum += math.exp(value - running_max)

        row_out = torch.zeros_like(row)
        if running_sum > 0.0:
            for i, (item, keep) in enumerate(zip(row, row_mask)):
                if bool(keep):
                    row_out[i] = math.exp(float(item) - running_max) / running_sum
        out_rows.append(row_out)
    return torch.stack(out_rows, dim=0)


def causal_softmax_naive(scores: torch.Tensor) -> torch.Tensor:
    # Causal softmax for attention scores [B, T, T].
    bsz, t_q, t_k = scores.shape
    out = torch.zeros_like(scores)
    for b in range(bsz):
        for i in range(t_q):
            row = scores[b, i]
            mask = torch.arange(t_k, device=scores.device) <= i
            out[b, i] = masked_softmax_naive(row.unsqueeze(0), mask.unsqueeze(0))[0]
    return out


def logsumexp_naive(row: torch.Tensor) -> float:
    # Naive logsumexp for 1D tensor.
    total = 0.0
    for item in row:
        total += math.exp(float(item))
    return math.log(total)


def logsumexp_stable(row: torch.Tensor) -> float:
    # Stable logsumexp with max subtraction.
    row_max = float(torch.max(row))
    total = 0.0
    for item in row:
        total += math.exp(float(item) - row_max)
    return row_max + math.log(total)


def softmax_backward_naive(softmax_out: torch.Tensor, grad_out: torch.Tensor) -> torch.Tensor:
    # Naive Jacobian-based backward for row-wise softmax on [M, N].
    grad_in_rows = []
    for y, g in zip(softmax_out, grad_out):
        n = y.numel()
        jacobian = torch.zeros((n, n), dtype=y.dtype, device=y.device)
        for i in range(n):
            for j in range(n):
                jacobian[i, j] = y[i] * ((1.0 if i == j else 0.0) - y[j])
        grad_in_rows.append(jacobian @ g)
    return torch.stack(grad_in_rows, dim=0)


def softmax_backward_fused(softmax_out: torch.Tensor, grad_out: torch.Tensor) -> torch.Tensor:
    # Fused backward formula: y * (g - sum(g * y)).
    dot = torch.sum(grad_out * softmax_out, dim=-1, keepdim=True)
    return softmax_out * (grad_out - dot)


# -----------------------------
# Practice Questions
# -----------------------------
# Beginner:
# 1) Why does softmax_naive_unsafe overflow for large logits?
# 2) Why is subtracting row max mathematically valid in softmax_safe?
# 3) What does causal_softmax_naive prevent in autoregressive decoding?
# Intermediate:
# 4) Compare masked_softmax_naive vs masked_softmax_online for memory access and temporary tensors.
# 5) How would you extend softmax_online_rowwise to support padding masks in one pass?
# 6) Why does logsumexp_stable remain finite when logsumexp_naive overflows?
# Advanced:
# 7) Derive softmax_backward_fused from the Jacobian form.
# 8) In which cases can fused backward lose precision versus Jacobian accumulation?
# 9) How would you tile softmax for long sequence rows on GPU shared memory?
# Expert:
# 10) Design a block-wise online softmax kernel with warp-level reductions and explain synchronization points.
# 11) Explain how to fuse softmax with dropout while preserving reproducibility.
# 12) What changes are needed to support FP8 logits with FP16/FP32 accumulation?
