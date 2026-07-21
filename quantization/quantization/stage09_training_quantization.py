from __future__ import annotations
import torch
from torch.autograd import Function
from .stage01_int8_quantization import quantize_affine, dequantize_affine
from .constants import EPS


class FakeQuantizeSTE(Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, qmin, qmax):
        ctx.save_for_backward(x)
        ctx.qmin = qmin; ctx.qmax = qmax; ctx.scale = scale; ctx.zero_point = zero_point
        q = torch.round(x / scale + zero_point).clamp(qmin, qmax)
        return (q - zero_point) * scale

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        lo = (ctx.qmin - ctx.zero_point) * ctx.scale
        hi = (ctx.qmax - ctx.zero_point) * ctx.scale
        mask = (x >= lo) & (x <= hi)
        return grad_out * mask.to(grad_out.dtype), None, None, None, None


def fake_quantize(x: torch.Tensor, scale: torch.Tensor | float, zero_point: int | torch.Tensor = 0,
                  qmin: int = -128, qmax: int = 127) -> torch.Tensor:
    return FakeQuantizeSTE.apply(x, torch.as_tensor(scale, device=x.device), torch.as_tensor(zero_point, device=x.device), qmin, qmax)


def qat_forward_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
                       x_scale: float, w_scale: float) -> torch.Tensor:
    x_fq = fake_quantize(x, x_scale, 0)
    w_fq = fake_quantize(weight, w_scale, 0)
    return torch.nn.functional.linear(x_fq, w_fq, bias)


def quantize_ste_backward(grad_out: torch.Tensor, x: torch.Tensor, clip_min: float, clip_max: float) -> torch.Tensor:
    return grad_out * ((x >= clip_min) & (x <= clip_max)).to(grad_out.dtype)


def scale_gradient_reference(grad_out: torch.Tensor, num_elements: int) -> torch.Tensor:
    if num_elements <= 0:
        raise ValueError("num_elements must be positive")
    return grad_out / (float(num_elements) ** 0.5)


def learned_scale_quantization(x: torch.Tensor, scale: torch.Tensor, qmin: int = -128, qmax: int = 127) -> torch.Tensor:
    s = torch.clamp(scale, min=EPS)
    q = torch.round((x / s).clamp(qmin, qmax))
    return q * s


def lsq_scale_grad_factor(x: torch.Tensor, qmax: int = 127) -> float:
    return 1.0 / ((x.numel() * qmax) ** 0.5)
