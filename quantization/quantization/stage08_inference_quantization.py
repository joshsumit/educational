from __future__ import annotations
import torch
from .stage01_int8_quantization import symmetric_int8_quantize_naive, symmetric_int8_dequantize
from .stage02_granularity import compute_per_channel_qparams, quantize_per_channel_int8, dequantize_per_channel_int8
from .stage03_calibration import minmax_calibration, percentile_calibration
from .stage07_quantized_gemm import int8_matmul_reference


def weight_only_quantization_reference(x: torch.Tensor, w: torch.Tensor):
    w_q, w_scale = symmetric_int8_quantize_naive(w)
    out = x @ symmetric_int8_dequantize(w_q, w_scale)
    return out, w_q, w_scale


def activation_quantization_reference(x: torch.Tensor):
    return symmetric_int8_quantize_naive(x)


def dynamic_activation_quantization(x: torch.Tensor):
    q, s = symmetric_int8_quantize_naive(x)
    return q, s, symmetric_int8_dequantize(q, s)


def quantize_linear_layer(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
                          per_output_channel: bool = True) -> torch.Tensor:
    x_q, x_s = symmetric_int8_quantize_naive(x)
    if per_output_channel:
        # weight shape: out_features, in_features; b for x @ weight.T is in_features x out_features
        scales, zps = compute_per_channel_qparams(weight, axis=0, symmetric=True)
        w_q = quantize_per_channel_int8(weight, scales, zps, axis=0)
        w_dq = dequantize_per_channel_int8(w_q, scales, zps, axis=0)
        out = x @ w_dq.t()
    else:
        w_q, w_s = symmetric_int8_quantize_naive(weight)
        out = int8_matmul_reference(x_q, w_q.t(), float(x_s), float(w_s))
    return out if bias is None else out + bias


def quantize_conv2d_layer(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
                          stride=1, padding=0, dilation=1, groups=1) -> torch.Tensor:
    scales, zps = compute_per_channel_qparams(weight, axis=0, symmetric=True)
    w_q = quantize_per_channel_int8(weight, scales, zps, axis=0)
    w_dq = dequantize_per_channel_int8(w_q, scales, zps, axis=0)
    x_q, x_s = symmetric_int8_quantize_naive(x)
    x_dq = symmetric_int8_dequantize(x_q, x_s)
    return torch.nn.functional.conv2d(x_dq, w_dq, bias, stride, padding, dilation, groups)


def offline_calibration(samples: list[torch.Tensor], method: str = "minmax"):
    x = torch.cat([s.flatten().to(torch.float32) for s in samples])
    if method == "minmax":
        return minmax_calibration(x)
    if method == "percentile":
        return percentile_calibration(x)
    raise ValueError(f"unknown calibration method: {method}")
