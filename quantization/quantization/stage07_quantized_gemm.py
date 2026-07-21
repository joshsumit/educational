from __future__ import annotations
import torch
from .stage01_int8_quantization import dequantize_affine
from .stage06_quantized_arithmetic import requantize_float_multiplier


def int8_matmul_reference(a_q: torch.Tensor, b_q: torch.Tensor, a_scale: float, b_scale: float,
                          a_zero_point: int = 0, b_zero_point: int = 0) -> torch.Tensor:
    a = dequantize_affine(a_q, a_scale, a_zero_point)
    b = dequantize_affine(b_q, b_scale, b_zero_point)
    return a @ b


def true_int8_gemm_int32_naive(a_q: torch.Tensor, b_q: torch.Tensor, a_zero_point: int = 0,
                               b_zero_point: int = 0) -> torch.Tensor:
    if a_q.dtype != torch.int8 or b_q.dtype != torch.int8:
        raise ValueError("a_q and b_q must be int8")
    a = a_q.to(torch.int32) - int(a_zero_point)
    b = b_q.to(torch.int32) - int(b_zero_point)
    return a @ b


def int8_gemm_dequantize(out_int32: torch.Tensor, a_scale: float, b_scale: float) -> torch.Tensor:
    return out_int32.to(torch.float32) * (float(a_scale) * float(b_scale))


def gemm_zero_point_correction(a_q: torch.Tensor, b_q: torch.Tensor, a_zero_point: int, b_zero_point: int) -> torch.Tensor:
    k = a_q.shape[-1]
    raw = a_q.to(torch.int32) @ b_q.to(torch.int32)
    row_sum = torch.sum(a_q.to(torch.int32), dim=-1, keepdim=True)
    col_sum = torch.sum(b_q.to(torch.int32), dim=-2, keepdim=True)
    return raw - int(b_zero_point) * row_sum - int(a_zero_point) * col_sum + k * int(a_zero_point) * int(b_zero_point)


def fixed_point_matmul_int8(a_q: torch.Tensor, b_q: torch.Tensor, a_scale: float, b_scale: float, out_scale: float,
                            a_zero_point: int = 0, b_zero_point: int = 0, out_zero_point: int = 0):
    acc = true_int8_gemm_int32_naive(a_q, b_q, a_zero_point, b_zero_point)
    q_out = requantize_float_multiplier(acc, (a_scale * b_scale) / out_scale, out_zero_point)
    return q_out, acc


def int4_gemm_reference(a_q4: torch.Tensor, b_q4: torch.Tensor, a_scale: float, b_scale: float) -> torch.Tensor:
    acc = a_q4.to(torch.int32) @ b_q4.to(torch.int32)
    return acc.to(torch.float32) * (a_scale * b_scale)


def gemm_bias_add(acc_or_float: torch.Tensor, bias: torch.Tensor, output_is_int32: bool = False,
                  acc_scale: float | None = None) -> torch.Tensor:
    if output_is_int32:
        if acc_scale is None:
            raise ValueError("acc_scale is required when adding float bias to int32 accumulators")
        return acc_or_float.to(torch.float32) * acc_scale + bias
    return acc_or_float + bias


def gemm_activation(x: torch.Tensor, kind: str = "relu") -> torch.Tensor:
    if kind == "identity":
        return x
    if kind == "relu":
        return torch.clamp(x, min=0)
    if kind == "gelu":
        return torch.nn.functional.gelu(x)
    raise ValueError(f"unknown activation: {kind}")
