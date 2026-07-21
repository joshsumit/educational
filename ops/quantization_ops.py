import torch

from .matmul_attention_ops import matmul_naive


# -----------------------------
# Basic quantization
# -----------------------------

def symmetric_int8_quantize_naive(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Quantize tensor to symmetric INT8 with zero-point fixed at 0."""
    max_abs = float(torch.max(torch.abs(x)))
    scale = max(max_abs / 127.0, 1e-12)
    q = torch.round(x / scale).clamp(-128, 127).to(torch.int8)
    return q, scale


def symmetric_int8_dequantize(q: torch.Tensor, scale: float) -> torch.Tensor:
    """Dequantize symmetric INT8 tensor back to float32."""
    return q.to(torch.float32) * float(scale)


def asymmetric_int8_quantize_naive(x: torch.Tensor) -> tuple[torch.Tensor, float, int]:
    """Quantize tensor to asymmetric INT8 with learned scale and zero-point."""
    x_min = float(torch.min(x))
    x_max = float(torch.max(x))
    scale = max((x_max - x_min) / 255.0, 1e-12)
    zero_point = int(round(-x_min / scale - 128.0))
    zero_point = max(-128, min(127, zero_point))
    q = torch.round(x / scale + zero_point).clamp(-128, 127).to(torch.int8)
    return q, scale, zero_point


def asymmetric_int8_dequantize(q: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    """Dequantize asymmetric INT8 tensor back to float32."""
    return (q.to(torch.float32) - float(zero_point)) * float(scale)


def dynamic_quantization_int8(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Apply dynamic INT8 quantization by deriving scale from current tensor."""
    return symmetric_int8_quantize_naive(x)


def static_quantization_int8(x: torch.Tensor, scale: float, zero_point: int = 0, symmetric: bool = True) -> torch.Tensor:
    """Apply static INT8 quantization using precomputed calibration parameters."""
    if symmetric:
        return torch.round(x / scale).clamp(-128, 127).to(torch.int8)
    return torch.round(x / scale + zero_point).clamp(-128, 127).to(torch.int8)


def quantize_per_tensor_int8(x: torch.Tensor, scale: float, zero_point: int = 0) -> torch.Tensor:
    """Quantize with per-tensor affine INT8 parameters."""
    q = torch.round(x / scale + zero_point)
    q = torch.clamp(q, -128, 127)
    return q.to(torch.int8)


def dequantize_per_tensor_int8(q: torch.Tensor, scale: float, zero_point: int = 0) -> torch.Tensor:
    """Dequantize per-tensor affine INT8 values to float32."""
    return (q.to(torch.float32) - float(zero_point)) * float(scale)


def quantize_per_channel_int8(x: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """Quantize with per-channel affine INT8 parameters along the given axis."""
    view_shape = [1] * x.dim()
    view_shape[axis] = x.shape[axis]
    s = scales.view(view_shape)
    z = zero_points.view(view_shape)
    q = torch.round(x / s + z)
    q = torch.clamp(q, -128, 127)
    return q.to(torch.int8)


def dequantize_per_channel_int8(q: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """Dequantize per-channel affine INT8 values to float32 along the given axis."""
    view_shape = [1] * q.dim()
    view_shape[axis] = q.shape[axis]
    s = scales.view(view_shape)
    z = zero_points.view(view_shape)
    return (q.to(torch.float32) - z.to(torch.float32)) * s.to(torch.float32)


def per_group_quantization_int8(x: torch.Tensor, group_size: int, axis: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize per group along an axis and return INT8 values plus group scales."""
    x_perm = x.movedim(axis, -1)
    n = x_perm.shape[-1]
    n_groups = (n + group_size - 1) // group_size

    q = torch.empty_like(x_perm, dtype=torch.int8)
    scales = torch.empty((*x_perm.shape[:-1], n_groups), dtype=torch.float32, device=x.device)

    for g in range(n_groups):
        s = g * group_size
        e = min((g + 1) * group_size, n)
        chunk = x_perm[..., s:e]
        max_abs = torch.max(torch.abs(chunk), dim=-1).values
        scale = torch.clamp(max_abs / 127.0, min=1e-12)
        scales[..., g] = scale

        scale_expanded = scale.unsqueeze(-1)
        q[..., s:e] = torch.round(chunk / scale_expanded).clamp(-128, 127).to(torch.int8)

    return q.movedim(-1, axis), scales


def per_block_quantization_int8(x: torch.Tensor, block_shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize each 2D block independently and return INT8 values plus block scales."""
    if x.dim() != 2:
        raise ValueError("per_block_quantization_int8 expects 2D tensor")

    h, w = x.shape
    bh, bw = block_shape
    q = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty(((h + bh - 1) // bh, (w + bw - 1) // bw), dtype=torch.float32, device=x.device)

    for bi, hs in enumerate(range(0, h, bh)):
        for bj, ws in enumerate(range(0, w, bw)):
            he = min(hs + bh, h)
            we = min(ws + bw, w)
            block = x[hs:he, ws:we]
            max_abs = float(torch.max(torch.abs(block)))
            scale = max(max_abs / 127.0, 1e-12)
            scales[bi, bj] = scale
            q[hs:he, ws:we] = torch.round(block / scale).clamp(-128, 127).to(torch.int8)

    return q, scales


# -----------------------------
# Low-bit quantization
# -----------------------------

def int4_quantize_naive(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Quantize to signed INT4 stored in INT8 container, range [-8, 7]."""
    return torch.round(x / scale).clamp(-8, 7).to(torch.int8)


def uint4_quantize_naive(x: torch.Tensor, scale: float, zero_point: int = 8) -> torch.Tensor:
    """Quantize to unsigned INT4 stored in UINT8 container, range [0, 15]."""
    q = torch.round(x / scale + zero_point).clamp(0, 15)
    return q.to(torch.uint8)


def int2_quantize_naive(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Quantize to signed INT2 stored in INT8 container, range [-2, 1]."""
    return torch.round(x / scale).clamp(-2, 1).to(torch.int8)


def nf4_quantize_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize to NF4 indices using a fixed 16-value codebook reference."""
    codebook = torch.tensor(
        [-1.0, -0.696, -0.525, -0.394, -0.284, -0.184, -0.089, 0.0, 0.079, 0.160, 0.246, 0.338, 0.441, 0.562, 0.723, 1.0],
        dtype=torch.float32,
        device=x.device,
    )
    max_abs = torch.clamp(torch.max(torch.abs(x)), min=1e-12)
    x_n = x / max_abs

    # Brute-force nearest codebook index.
    flat = x_n.reshape(-1, 1)
    dists = torch.abs(flat - codebook.view(1, -1))
    idx = torch.argmin(dists, dim=1).to(torch.uint8)
    return idx.reshape(x.shape), codebook * max_abs


def nf4_dequantize_reference(indices: torch.Tensor, scaled_codebook: torch.Tensor) -> torch.Tensor:
    """Dequantize NF4 indices through a scaled codebook lookup."""
    return scaled_codebook[indices.to(torch.int64)]


# -----------------------------
# Floating-point quantization
# -----------------------------

def fp16_conversion(x: torch.Tensor) -> torch.Tensor:
    """Convert tensor to FP16 precision."""
    return x.to(torch.float16)


def bf16_conversion(x: torch.Tensor) -> torch.Tensor:
    """Convert tensor to BF16 precision."""
    return x.to(torch.bfloat16)


def _quantize_fp_like(x: torch.Tensor, exp_bits: int, mant_bits: int) -> torch.Tensor:
    """Simulate reduced-precision FP formats by quantizing exponent and mantissa."""
    x_f = x.to(torch.float32)
    sign = torch.sign(x_f)
    abs_x = torch.abs(x_f)

    # Handle zero separately to avoid log2(0).
    non_zero = abs_x > 0
    out = torch.zeros_like(x_f)

    if torch.any(non_zero):
        nz = abs_x[non_zero]
        exp = torch.floor(torch.log2(nz))
        mant = nz / torch.pow(2.0, exp) - 1.0

        mant_q = torch.round(mant * (2**mant_bits)) / (2**mant_bits)

        # Bias and exponent clamping to emulate limited exponent field.
        bias = 2 ** (exp_bits - 1) - 1
        exp_min = -bias
        exp_max = bias
        exp_q = torch.clamp(exp, exp_min, exp_max)

        nz_q = (1.0 + mant_q) * torch.pow(2.0, exp_q)
        out[non_zero] = nz_q

    return out * sign


def fp8_e4m3_conversion_reference(x: torch.Tensor) -> torch.Tensor:
    """Simulate FP8 E4M3 conversion with a software reference path."""
    return _quantize_fp_like(x, exp_bits=4, mant_bits=3)


def fp8_e5m2_conversion_reference(x: torch.Tensor) -> torch.Tensor:
    """Simulate FP8 E5M2 conversion with a software reference path."""
    return _quantize_fp_like(x, exp_bits=5, mant_bits=2)


# -----------------------------
# Quantized compute
# -----------------------------

def int8_matmul_reference(
    a_q: torch.Tensor,
    b_q: torch.Tensor,
    a_scale: float,
    b_scale: float,
    a_zero_point: int = 0,
    b_zero_point: int = 0,
) -> torch.Tensor:
    """Run reference INT8 matmul by dequantizing inputs to float32 first."""
    a_f = dequantize_per_tensor_int8(a_q, a_scale, a_zero_point)
    b_f = dequantize_per_tensor_int8(b_q, b_scale, b_zero_point)
    return matmul_naive(a_f, b_f).to(torch.float32)


def true_int8_gemm_int32_naive(
    a_q: torch.Tensor,
    b_q: torch.Tensor,
    a_zero_point: int = 0,
    b_zero_point: int = 0,
) -> torch.Tensor:
    """Run true INT8 GEMM with INT32 accumulation output."""
    if a_q.dtype != torch.int8 or b_q.dtype != torch.int8:
        raise ValueError("Inputs must be int8")

    m, k_a = a_q.shape
    k_b, n = b_q.shape
    if k_a != k_b:
        raise ValueError("Inner dimensions mismatch")

    out = torch.zeros((m, n), dtype=torch.int32, device=a_q.device)
    for i in range(m):
        for j in range(n):
            acc = 0
            for k in range(k_a):
                av = int(a_q[i, k]) - a_zero_point
                bv = int(b_q[k, j]) - b_zero_point
                acc += av * bv
            out[i, j] = acc
    return out


def int8_gemm_dequantize(out_int32: torch.Tensor, a_scale: float, b_scale: float) -> torch.Tensor:
    """Dequantize INT32 GEMM accumulators to float32."""
    return out_int32.to(torch.float32) * (a_scale * b_scale)


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
    """Run INT8 GEMM, keep INT32 accumulators, and requantize to INT8 output."""
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


def int4_gemm_reference(a_q4: torch.Tensor, b_q4: torch.Tensor, a_scale: float, b_scale: float) -> torch.Tensor:
    """Run INT4 GEMM reference with INT32 accumulation and float dequantization."""
    a_i = a_q4.to(torch.int32)
    b_i = b_q4.to(torch.int32)
    out_i32 = a_i @ b_i
    return out_i32.to(torch.float32) * (a_scale * b_scale)


def weight_only_quantization_reference(x: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Reference weight-only quantization: quantize weights and keep activations in float."""
    w_q, w_scale = symmetric_int8_quantize_naive(w)
    out = x @ symmetric_int8_dequantize(w_q, w_scale)
    return out, w_q, w_scale


def activation_quantization_reference(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Reference activation-only dynamic INT8 quantization."""
    return dynamic_quantization_int8(x)


def gptq_style_quantization_reference(w: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Simplified GPTQ-style per-group weight quantization layout reference."""
    q, scales = per_group_quantization_int8(w, group_size=group_size, axis=-1)
    return q, scales


def awq_style_quantization_reference(w: torch.Tensor, act_samples: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simplified AWQ-style quantization with activation-derived channel importance."""
    importance = torch.mean(torch.abs(act_samples), dim=0)
    importance = importance / torch.clamp(torch.max(importance), min=1e-12)

    # Scale weights before quantization to protect important channels.
    w_scaled = w * (1.0 + importance.unsqueeze(0))
    q, scales = per_group_quantization_int8(w_scaled, group_size=group_size, axis=-1)
    return q, scales, importance


def quantize_ste_backward(grad_out: torch.Tensor, x: torch.Tensor, clip_min: float, clip_max: float) -> torch.Tensor:
    """Straight-through estimator for fake-quant clipping windows."""
    mask = (x >= clip_min) & (x <= clip_max)
    return grad_out * mask.to(grad_out.dtype)


def scale_gradient_reference(grad_out: torch.Tensor, num_elements: int) -> torch.Tensor:
    """Scale gradient using a sqrt(N) normalization heuristic."""
    if num_elements <= 0:
        raise ValueError("num_elements must be positive")
    return grad_out / float(num_elements) ** 0.5


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
#
# Additional question set:
# Beginner:
# 1) What is the difference between symmetric and asymmetric INT8 quantization?
# 2) Why does dynamic quantization choose scale at runtime?
# 3) What is the tradeoff between FP16 and BF16?
# Intermediate:
# 4) Compare per-group and per-block quantization granularity.
# 5) Why might NF4 preserve quality better than uniform INT4?
# 6) How do zero-points affect integer arithmetic in GEMM kernels?
# Advanced:
# 7) Explain true INT8 GEMM with INT32 accumulation and output requantization.
# 8) How would you calibrate static quantization for long-tail activations?
# 9) Discuss E4M3 vs E5M2 exponent/mantissa tradeoffs for FP8.
# Expert:
# 10) Outline a GPTQ pipeline with Hessian approximation and blockwise solve.
# 11) How does AWQ protect salient channels, and when can it fail?
# 12) Design an evaluation protocol for latency-accuracy Pareto across quantization schemes.
