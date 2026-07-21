import math
import torch


def layernorm_naive(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # Basic LayerNorm over last dimension.
    mean = torch.mean(x, dim=-1, keepdim=True)
    var = torch.mean((x - mean) ** 2, dim=-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps)


def layernorm_two_pass(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # Explicit two-pass LayerNorm reference for learning kernel decomposition.
    out_rows = []
    for row in x:
        values = [float(v) for v in row]
        mean = sum(values) / len(values)

        var_acc = 0.0
        for value in values:
            diff = value - mean
            var_acc += diff * diff
        var = var_acc / len(values)

        inv_std = 1.0 / math.sqrt(var + eps)
        out_rows.append(torch.tensor([(value - mean) * inv_std for value in values], dtype=x.dtype, device=x.device))
    return torch.stack(out_rows, dim=0)


def rmsnorm_naive(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # RMSNorm over last dimension without mean subtraction.
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms


def rmsnorm_two_pass(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # Explicit RMSNorm reference with scalar loops per row.
    out_rows = []
    for row in x:
        values = [float(v) for v in row]
        sq_acc = 0.0
        for value in values:
            sq_acc += value * value
        mean_sq = sq_acc / len(values)
        inv_rms = 1.0 / math.sqrt(mean_sq + eps)
        out_rows.append(torch.tensor([value * inv_rms for value in values], dtype=x.dtype, device=x.device))
    return torch.stack(out_rows, dim=0)


def gelu_exact(x: torch.Tensor) -> torch.Tensor:
    # Exact GELU using error function.
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_tanh_approx(x: torch.Tensor) -> torch.Tensor:
    # Fast tanh GELU approximation used in many inference paths.
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x * x * x)))


def silu(x: torch.Tensor) -> torch.Tensor:
    # SiLU / Swish activation.
    return x * torch.sigmoid(x)


# -----------------------------
# Practice Questions
# -----------------------------
# Beginner:
# 1) What is the difference between LayerNorm and RMSNorm mathematically?
# 2) Why does LayerNorm include epsilon?
# 3) Compare GELU and SiLU behavior for negative inputs.
# Intermediate:
# 4) Why is layernorm_two_pass useful to understand kernel decomposition?
# 5) When is gelu_tanh_approx preferred over gelu_exact?
# 6) How do normalization ops affect gradient scaling in deep transformers?
# Advanced:
# 7) How would you compute LayerNorm statistics using Welford in one pass?
# 8) Which operations can be fused with LayerNorm in inference kernels?
# 9) How would you validate numerical drift between FP32 and BF16 normalization?
# Expert:
# 10) Design a warp-level LayerNorm kernel for hidden size 4096.
# 11) How do you avoid bank conflicts in reduction over hidden dimension?
# 12) What are tradeoffs of persistent-kernel normalization in serving?
