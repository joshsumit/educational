from __future__ import annotations
import torch
from .constants import EPS


def fp16_conversion(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float16)


def bf16_conversion(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16)


def fp32_to_bf16_bits(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32).view(torch.int32).bitwise_right_shift(16).to(torch.int16)


def _quantize_fp_like(x: torch.Tensor, exp_bits: int, mant_bits: int, exp_bias: int | None = None) -> torch.Tensor:
    xf = x.to(torch.float32)
    sign = torch.sign(xf)
    ax = torch.abs(xf)
    if exp_bias is None:
        exp_bias = (1 << (exp_bits - 1)) - 1
    max_exp = (1 << exp_bits) - 2 - exp_bias
    min_exp = 1 - exp_bias
    normal = ax >= (2.0 ** min_exp)
    safe = torch.clamp(ax, min=EPS)
    exp = torch.floor(torch.log2(safe)).clamp(min_exp, max_exp)
    step = torch.pow(torch.tensor(2.0, device=x.device), exp - mant_bits)
    q = torch.round(ax / step) * step
    max_val = (2.0 - 2.0 ** (-mant_bits)) * (2.0 ** max_exp)
    q = torch.clamp(q, 0.0, max_val)
    # crude subnormal support: constant step at minimum exponent
    sub_step = 2.0 ** (min_exp - mant_bits)
    q_sub = torch.round(ax / sub_step) * sub_step
    q = torch.where(normal, q, q_sub)
    q = torch.where(torch.isfinite(xf), q, ax)
    return q * sign


def fp8_e4m3_conversion_reference(x: torch.Tensor) -> torch.Tensor:
    return _quantize_fp_like(x, exp_bits=4, mant_bits=3, exp_bias=7)


def fp8_e5m2_conversion_reference(x: torch.Tensor) -> torch.Tensor:
    return _quantize_fp_like(x, exp_bits=5, mant_bits=2, exp_bias=15)


def fp8_quantize_scaled(x: torch.Tensor, fmt: str = "e4m3"):
    amax = torch.clamp(torch.max(torch.abs(x.to(torch.float32))), min=EPS)
    max_fp8 = 240.0 if fmt.lower() == "e4m3" else 57344.0
    scale = amax / max_fp8
    y = x / scale
    q = fp8_e4m3_conversion_reference(y) if fmt.lower() == "e4m3" else fp8_e5m2_conversion_reference(y)
    return q, scale


def fp8_dequantize_scaled(q: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
    return q.to(torch.float32) * torch.as_tensor(scale, device=q.device, dtype=torch.float32)
