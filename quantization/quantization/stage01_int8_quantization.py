from __future__ import annotations
import torch
from .stage00_foundations import asymmetric_scale_zero_point, symmetric_scale
from .stage00_foundations import integer_range
from .constants import EPS


def quantize_affine(x: torch.Tensor, scale: torch.Tensor | float, zero_point: torch.Tensor | int = 0,
                    num_bits: int = 8, signed: bool = True) -> torch.Tensor:
    qmin, qmax = integer_range(num_bits, signed)
    q = torch.round(x.to(torch.float32) / torch.as_tensor(scale, device=x.device, dtype=torch.float32) +
                    torch.as_tensor(zero_point, device=x.device, dtype=torch.float32))
    dtype = torch.int8 if signed and num_bits <= 8 else torch.uint8 if not signed and num_bits <= 8 else torch.int32
    return torch.clamp(q, qmin, qmax).to(dtype)


def dequantize_affine(q: torch.Tensor, scale: torch.Tensor | float, zero_point: torch.Tensor | int = 0) -> torch.Tensor:
    return (q.to(torch.float32) - torch.as_tensor(zero_point, device=q.device, dtype=torch.float32)) * torch.as_tensor(scale, device=q.device, dtype=torch.float32)


def symmetric_int8_quantize_naive(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = symmetric_scale(x, num_bits=8, signed=True)
    return quantize_affine(x, scale, 0, 8, True), scale


def symmetric_int8_dequantize(q: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
    return dequantize_affine(q, scale, 0)


def asymmetric_int8_quantize_naive(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale, zp = asymmetric_scale_zero_point(x, 8, True)
    return quantize_affine(x, scale, zp, 8, True), scale, zp


def asymmetric_int8_dequantize(q: torch.Tensor, scale: torch.Tensor | float, zero_point: torch.Tensor | int) -> torch.Tensor:
    return dequantize_affine(q, scale, zero_point)


def dynamic_quantization_int8(x: torch.Tensor, symmetric: bool = True):
    return symmetric_int8_quantize_naive(x) if symmetric else asymmetric_int8_quantize_naive(x)


def static_quantization_int8(x: torch.Tensor, scale: float | torch.Tensor, zero_point: int | torch.Tensor = 0,
                             symmetric: bool = True) -> torch.Tensor:
    return quantize_affine(x, scale, 0 if symmetric else zero_point, 8, True)
