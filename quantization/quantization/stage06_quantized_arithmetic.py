from __future__ import annotations
import math
import torch
from .stage01_int8_quantization import dequantize_affine, quantize_affine


def requantize_float_multiplier(acc: torch.Tensor, multiplier: float | torch.Tensor, out_zero_point: int = 0,
                                qmin: int = -128, qmax: int = 127) -> torch.Tensor:
    q = torch.round(acc.to(torch.float32) * torch.as_tensor(multiplier, device=acc.device) + out_zero_point)
    return torch.clamp(q, qmin, qmax).to(torch.int8)


def fixed_point_multiplier(real_multiplier: float) -> tuple[int, int]:
    if real_multiplier <= 0:
        raise ValueError("real_multiplier must be positive")
    significand, exponent = math.frexp(real_multiplier)
    q31 = int(round(significand * (1 << 31)))
    if q31 == (1 << 31):
        q31 //= 2
        exponent += 1
    return q31, exponent


def requantize_fixed_point(acc: torch.Tensor, q31_multiplier: int, exponent: int, out_zero_point: int = 0,
                           qmin: int = -128, qmax: int = 127) -> torch.Tensor:
    x = acc.to(torch.int64) * int(q31_multiplier)
    x = (x + (1 << 30)) >> 31
    if exponent >= 0:
        x = x << exponent
    else:
        r = -exponent
        x = (x + (1 << (r - 1))) >> r
    x = x + int(out_zero_point)
    return torch.clamp(x, qmin, qmax).to(torch.int8)


def integer_add_same_qparams(a_q: torch.Tensor, b_q: torch.Tensor, zero_point: int = 0,
                             qmin: int = -128, qmax: int = 127) -> torch.Tensor:
    out = a_q.to(torch.int32) + b_q.to(torch.int32) - int(zero_point)
    return torch.clamp(out, qmin, qmax).to(torch.int8)


def integer_mul_same_qparams(a_q: torch.Tensor, b_q: torch.Tensor, scale: float, zero_point: int = 0,
                             out_scale: float | None = None) -> torch.Tensor:
    out_scale = scale if out_scale is None else out_scale
    acc = (a_q.to(torch.int32) - zero_point) * (b_q.to(torch.int32) - zero_point)
    return requantize_float_multiplier(acc, (scale * scale) / out_scale, zero_point)


def fake_quantize_reference(x: torch.Tensor, scale: float, zero_point: int = 0, qmin: int = -128, qmax: int = 127):
    q = torch.round(x / scale + zero_point).clamp(qmin, qmax)
    return (q - zero_point) * scale
