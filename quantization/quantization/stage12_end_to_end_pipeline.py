from __future__ import annotations
from dataclasses import dataclass
import torch
from .stage01_int8_quantization import symmetric_int8_quantize_naive, symmetric_int8_dequantize
from .stage02_granularity import compute_per_channel_qparams, quantize_per_channel_int8, dequantize_per_channel_int8
from .stage03_calibration import percentile_calibration
from .stage07_quantized_gemm import true_int8_gemm_int32_naive, int8_gemm_dequantize


@dataclass
class QuantTensor:
    q: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor | int = 0
    axis: int | None = None

    def dequantize(self) -> torch.Tensor:
        if self.axis is None:
            return symmetric_int8_dequantize(self.q, self.scale)
        shape = [1] * self.q.dim(); shape[self.axis] = -1
        return (self.q.to(torch.float32) - torch.as_tensor(self.zero_point, device=self.q.device).view(shape)) * self.scale.view(shape)


@dataclass
class QuantizedLinear:
    weight: QuantTensor
    bias: torch.Tensor | None

    @staticmethod
    def from_float(weight: torch.Tensor, bias: torch.Tensor | None = None, per_channel: bool = True):
        if per_channel:
            s, z = compute_per_channel_qparams(weight, axis=0, symmetric=True)
            q = quantize_per_channel_int8(weight, s, z, axis=0)
            return QuantizedLinear(QuantTensor(q, s, z, axis=0), bias)
        q, s = symmetric_int8_quantize_naive(weight)
        return QuantizedLinear(QuantTensor(q, s, 0, None), bias)

    def forward_reference(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.dequantize()
        out = x @ w.t()
        return out if self.bias is None else out + self.bias


def ptq_pipeline_linear(x_calib: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
                        activation_percentile: float = 99.99) -> torch.Tensor:
    lo, hi = percentile_calibration(x_calib, activation_percentile, symmetric=True)
    act_scale = torch.clamp(torch.max(torch.abs(torch.stack([lo, hi]))) / 127.0, min=1e-12)
    x_q = torch.round(x / act_scale).clamp(-128, 127).to(torch.int8)
    qlinear = QuantizedLinear.from_float(weight, bias, per_channel=True)
    # Reference path dequantizes per-channel weights. Real kernels would consume int8 + scales directly.
    out = x_q.to(torch.float32) * act_scale @ qlinear.weight.dequantize().t()
    return out if bias is None else out + bias


def end_to_end_int8_gemm_demo(a: torch.Tensor, b: torch.Tensor):
    a_q, a_s = symmetric_int8_quantize_naive(a)
    b_q, b_s = symmetric_int8_quantize_naive(b)
    acc = true_int8_gemm_int32_naive(a_q, b_q)
    out = int8_gemm_dequantize(acc, float(a_s), float(b_s))
    return {"a_q": a_q, "b_q": b_q, "a_scale": a_s, "b_scale": b_s, "acc_int32": acc, "out_float": out}
