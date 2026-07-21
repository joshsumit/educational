import torch

from .matmul_attention_ops import matmul_naive


def quantize_per_tensor_int8(x: torch.Tensor, scale: float, zero_point: int = 0) -> torch.Tensor:
    # Per-tensor affine int8 quantization.
    q = torch.round(x / scale + zero_point)
    q = torch.clamp(q, -128, 127)
    return q.to(torch.int8)


def dequantize_per_tensor_int8(q: torch.Tensor, scale: float, zero_point: int = 0) -> torch.Tensor:
    # Per-tensor affine int8 dequantization to float.
    return (q.to(torch.float32) - float(zero_point)) * float(scale)


def quantize_per_channel_int8(x: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, axis: int = 0) -> torch.Tensor:
    # Per-channel affine int8 quantization along selected axis.
    view_shape = [1] * x.dim()
    view_shape[axis] = x.shape[axis]
    s = scales.view(view_shape)
    z = zero_points.view(view_shape)
    q = torch.round(x / s + z)
    q = torch.clamp(q, -128, 127)
    return q.to(torch.int8)


def dequantize_per_channel_int8(q: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, axis: int = 0) -> torch.Tensor:
    # Per-channel affine int8 dequantization along selected axis.
    view_shape = [1] * q.dim()
    view_shape[axis] = q.shape[axis]
    s = scales.view(view_shape)
    z = zero_points.view(view_shape)
    return (q.to(torch.float32) - z.to(torch.float32)) * s.to(torch.float32)


def int8_matmul_reference(
    a_q: torch.Tensor,
    b_q: torch.Tensor,
    a_scale: float,
    b_scale: float,
    a_zero_point: int = 0,
    b_zero_point: int = 0,
) -> torch.Tensor:
    # Reference int8 matmul with dequantized output in float32.
    # Inputs: a_q [M, K], b_q [K, N] as int8 tensors.
    a_f = dequantize_per_tensor_int8(a_q, a_scale, a_zero_point)
    b_f = dequantize_per_tensor_int8(b_q, b_scale, b_zero_point)
    return matmul_naive(a_f, b_f).to(torch.float32)


def fixed_point_matmul_int8(
    a_q: torch.Tensor,
    b_q: torch.Tensor,
    a_scale: float,
    b_scale: float,
    out_scale: float,
    a_zero_point: int = 0,
    b_zero_point: int = 0,
    out_zero_point: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Fixed-point style INT8 GEMM reference.
    # 1) accumulate in INT32, 2) requantize to INT8 output domain.
    if a_q.dtype != torch.int8 or b_q.dtype != torch.int8:
        raise ValueError("a_q and b_q must be int8")
    if out_scale <= 0 or a_scale <= 0 or b_scale <= 0:
        raise ValueError("scales must be positive")

    m, k_a = a_q.shape
    k_b, n = b_q.shape
    if k_a != k_b:
        raise ValueError("Inner dimensions must match")

    acc = torch.zeros((m, n), dtype=torch.int32, device=a_q.device)
    for i in range(m):
        for j in range(n):
            total = 0
            for k in range(k_a):
                av = int(a_q[i, k]) - a_zero_point
                bv = int(b_q[k, j]) - b_zero_point
                total += av * bv
            acc[i, j] = total

    requant_scale = (a_scale * b_scale) / out_scale
    out_q = torch.round(acc.to(torch.float32) * requant_scale + out_zero_point)
    out_q = torch.clamp(out_q, -128, 127).to(torch.int8)
    return out_q, acc


# -----------------------------
# Practice Questions
# -----------------------------
# Beginner:
# 1) What are scale and zero-point in affine quantization?
# 2) Why is dequantization needed before float kernels?
# 3) What error is introduced by rounding in INT8 quantization?
# Intermediate:
# 4) Why is per-channel quantization usually better than per-tensor for weights?
# 5) How do outliers impact chosen scale values?
# 6) What is the memory-bandwidth benefit of INT8 weights?
# Advanced:
# 7) How would you implement true INT8 GEMM with INT32 accumulation and requantization?
# 8) How do symmetric and asymmetric quantization differ in hardware cost?
# 9) How do you calibrate activation ranges for static quantization?
# Expert:
# 10) Design a fused quantized matmul + bias + activation kernel.
# 11) How do FP8 and INT8 compare for transformer inference quality/speed?
# 12) Explain quantization-aware training versus post-training quantization tradeoffs.
