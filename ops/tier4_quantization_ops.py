import math
import torch


# -----------------------------
# Basic quantization
# -----------------------------

def symmetric_int8_quantize_naive(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    # Symmetric INT8 quantization with zero-point fixed at 0.
    max_abs = float(torch.max(torch.abs(x)))
    scale = max(max_abs / 127.0, 1e-12)
    q = torch.round(x / scale).clamp(-128, 127).to(torch.int8)
    return q, scale


def symmetric_int8_dequantize(q: torch.Tensor, scale: float) -> torch.Tensor:
    # Dequantize symmetric INT8 tensor.
    return q.to(torch.float32) * float(scale)


def asymmetric_int8_quantize_naive(x: torch.Tensor) -> tuple[torch.Tensor, float, int]:
    # Asymmetric INT8 quantization with learned scale and zero-point.
    x_min = float(torch.min(x))
    x_max = float(torch.max(x))
    scale = max((x_max - x_min) / 255.0, 1e-12)
    zero_point = int(round(-x_min / scale - 128.0))
    zero_point = max(-128, min(127, zero_point))
    q = torch.round(x / scale + zero_point).clamp(-128, 127).to(torch.int8)
    return q, scale, zero_point


def asymmetric_int8_dequantize(q: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    # Dequantize asymmetric INT8 tensor.
    return (q.to(torch.float32) - float(zero_point)) * float(scale)


def dynamic_quantization_int8(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    # Dynamic quantization computes scale on the fly from current activation tensor.
    return symmetric_int8_quantize_naive(x)


def static_quantization_int8(x: torch.Tensor, scale: float, zero_point: int = 0, symmetric: bool = True) -> torch.Tensor:
    # Static quantization uses precomputed calibration parameters.
    if symmetric:
        return torch.round(x / scale).clamp(-128, 127).to(torch.int8)
    return torch.round(x / scale + zero_point).clamp(-128, 127).to(torch.int8)


def per_group_quantization_int8(x: torch.Tensor, group_size: int, axis: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    # Per-group quantization splits axis into groups and computes one scale per group.
    # Returns quantized tensor and scale tensor with group dimension.
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
    # Per-block quantization for 2D tensors.
    # block_shape=(bh,bw) defines local quantization regions.
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
    # Signed INT4 quantization stored in int8 container, range [-8, 7].
    return torch.round(x / scale).clamp(-8, 7).to(torch.int8)


def uint4_quantize_naive(x: torch.Tensor, scale: float, zero_point: int = 8) -> torch.Tensor:
    # Unsigned INT4 quantization stored in uint8 container, range [0, 15].
    q = torch.round(x / scale + zero_point).clamp(0, 15)
    return q.to(torch.uint8)


def int2_quantize_naive(x: torch.Tensor, scale: float) -> torch.Tensor:
    # Signed INT2 quantization stored in int8 container, range [-2, 1].
    return torch.round(x / scale).clamp(-2, 1).to(torch.int8)


def nf4_quantize_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # NF4 reference quantization using a fixed 16-value codebook.
    # This codebook is a simplified study reference.
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
    # Dequantize NF4 indices via scaled codebook lookup.
    return scaled_codebook[indices.to(torch.int64)]


# -----------------------------
# Floating-point quantization
# -----------------------------

def fp16_conversion(x: torch.Tensor) -> torch.Tensor:
    # Convert tensor to FP16.
    return x.to(torch.float16)


def bf16_conversion(x: torch.Tensor) -> torch.Tensor:
    # Convert tensor to BF16.
    return x.to(torch.bfloat16)


def _quantize_fp_like(x: torch.Tensor, exp_bits: int, mant_bits: int) -> torch.Tensor:
    # Educational FPx quantization simulation.
    # 1) decompose into sign, exponent, mantissa, 2) round mantissa to mant_bits, 3) clamp exponent range.
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
    # Simulated FP8 E4M3 conversion.
    return _quantize_fp_like(x, exp_bits=4, mant_bits=3)


def fp8_e5m2_conversion_reference(x: torch.Tensor) -> torch.Tensor:
    # Simulated FP8 E5M2 conversion.
    return _quantize_fp_like(x, exp_bits=5, mant_bits=2)


# -----------------------------
# Quantized compute
# -----------------------------

def true_int8_gemm_int32_naive(
    a_q: torch.Tensor,
    b_q: torch.Tensor,
    a_zero_point: int = 0,
    b_zero_point: int = 0,
) -> torch.Tensor:
    # True INT8 GEMM with INT32 accumulation.
    # a_q[M,K] int8, b_q[K,N] int8 -> out[M,N] int32.
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
    # Convert INT32 accumulators back to float domain.
    return out_int32.to(torch.float32) * (a_scale * b_scale)


def int4_gemm_reference(a_q4: torch.Tensor, b_q4: torch.Tensor, a_scale: float, b_scale: float) -> torch.Tensor:
    # INT4 GEMM reference by casting int4-in-int8 containers to int32 accumulation.
    a_i = a_q4.to(torch.int32)
    b_i = b_q4.to(torch.int32)
    out_i32 = a_i @ b_i
    return out_i32.to(torch.float32) * (a_scale * b_scale)


def weight_only_quantization_reference(x: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    # Weight-only quantization reference:
    # quantize weights, keep activations in float, then dequantized matmul.
    w_q, w_scale = symmetric_int8_quantize_naive(w)
    out = x @ symmetric_int8_dequantize(w_q, w_scale)
    return out, w_q, w_scale


def activation_quantization_reference(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    # Activation-only dynamic quantization reference.
    return dynamic_quantization_int8(x)


def gptq_style_quantization_reference(w: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    # GPTQ-style simplified reference:
    # per-group quantization intended to mimic second-order aware weight quantization layout.
    q, scales = per_group_quantization_int8(w, group_size=group_size, axis=-1)
    return q, scales


def awq_style_quantization_reference(w: torch.Tensor, act_samples: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # AWQ-style simplified reference:
    # estimate per-channel importance from activations, scale weights, then quantize.
    importance = torch.mean(torch.abs(act_samples), dim=0)
    importance = importance / torch.clamp(torch.max(importance), min=1e-12)

    # Scale weights before quantization to protect important channels.
    w_scaled = w * (1.0 + importance.unsqueeze(0))
    q, scales = per_group_quantization_int8(w_scaled, group_size=group_size, axis=-1)
    return q, scales, importance


def quantize_ste_backward(grad_out: torch.Tensor, x: torch.Tensor, clip_min: float, clip_max: float) -> torch.Tensor:
    # Straight-through estimator (STE) reference for fake-quant blocks.
    # Pass gradient through in unclipped range and block it outside the clamp window.
    mask = (x >= clip_min) & (x <= clip_max)
    return grad_out * mask.to(grad_out.dtype)


def scale_gradient_reference(grad_out: torch.Tensor, num_elements: int) -> torch.Tensor:
    # Common gradient scaling heuristic for learnable quantization scales.
    if num_elements <= 0:
        raise ValueError("num_elements must be positive")
    return grad_out / float(num_elements) ** 0.5


# -----------------------------
# Practice Questions
# -----------------------------
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
