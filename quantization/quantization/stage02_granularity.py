from __future__ import annotations
import torch
from .stage00_foundations import integer_range
from .stage01_int8_quantization import quantize_affine, dequantize_affine
from .constants import EPS


def _view_for_axis(v: torch.Tensor, dim: int, axis: int) -> torch.Tensor:
    axis = axis % dim
    shape = [1] * dim
    shape[axis] = -1
    return v.view(shape)


def compute_per_channel_qparams(x: torch.Tensor, axis: int = 0, num_bits: int = 8, symmetric: bool = True):
    qmin, qmax = integer_range(num_bits, True)
    axis = axis % x.dim()
    reduce_dims = [i for i in range(x.dim()) if i != axis]
    xf = x.to(torch.float32)
    if symmetric:
        max_abs = torch.amax(torch.abs(xf), dim=reduce_dims)
        scale = torch.clamp(max_abs / float(max(abs(qmin), abs(qmax))), min=EPS)
        zp = torch.zeros_like(scale, dtype=torch.int32)
    else:
        xmin = torch.amin(xf, dim=reduce_dims)
        xmax = torch.amax(xf, dim=reduce_dims)
        scale = torch.clamp((xmax - xmin) / float(qmax - qmin), min=EPS)
        zp = torch.round(qmin - xmin / scale).clamp(qmin, qmax).to(torch.int32)
    return scale, zp


def quantize_per_tensor_int8(x: torch.Tensor, scale: float | torch.Tensor, zero_point: int | torch.Tensor = 0) -> torch.Tensor:
    return quantize_affine(x, scale, zero_point, 8, True)


def dequantize_per_tensor_int8(q: torch.Tensor, scale: float | torch.Tensor, zero_point: int | torch.Tensor = 0) -> torch.Tensor:
    return dequantize_affine(q, scale, zero_point)


def quantize_per_channel_int8(x: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, axis: int = 0) -> torch.Tensor:
    return quantize_affine(x, _view_for_axis(scales, x.dim(), axis), _view_for_axis(zero_points, x.dim(), axis), 8, True)


def dequantize_per_channel_int8(q: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, axis: int = 0) -> torch.Tensor:
    return dequantize_affine(q, _view_for_axis(scales, q.dim(), axis), _view_for_axis(zero_points, q.dim(), axis))


def per_group_quantization_int8(x: torch.Tensor, group_size: int, axis: int = -1, symmetric: bool = True):
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    axis = axis % x.dim()
    xp = x.movedim(axis, -1).contiguous()
    n = xp.shape[-1]
    pad = (group_size - n % group_size) % group_size
    if pad:
        xp = torch.nn.functional.pad(xp, (0, pad))
    groups = xp.view(*xp.shape[:-1], -1, group_size)
    max_abs = torch.amax(torch.abs(groups.to(torch.float32)), dim=-1, keepdim=True)
    scale = torch.clamp(max_abs / 127.0, min=EPS)
    zp = torch.zeros_like(scale, dtype=torch.int32)
    qg = torch.round(groups / scale).clamp(-128, 127).to(torch.int8)
    q = qg.view(*xp.shape)[..., :n].movedim(-1, axis).contiguous()
    return q, scale.squeeze(-1).movedim(-1, axis if axis < x.dim()-1 else -1).contiguous()


def per_group_dequantization_int8(q: torch.Tensor, scales: torch.Tensor, group_size: int, axis: int = -1) -> torch.Tensor:
    axis = axis % q.dim()
    qp = q.movedim(axis, -1).contiguous()
    n = qp.shape[-1]
    pad = (group_size - n % group_size) % group_size
    if pad:
        qp = torch.nn.functional.pad(qp, (0, pad))
    groups = qp.view(*qp.shape[:-1], -1, group_size).to(torch.float32)
    sp = scales.movedim(axis if axis < scales.dim() else -1, -1).unsqueeze(-1).to(torch.float32)
    out = (groups * sp).view(*qp.shape)[..., :n]
    return out.movedim(-1, axis).contiguous()


def per_block_quantization_int8(x: torch.Tensor, block_shape: tuple[int, int]):
    if x.dim() != 2:
        raise ValueError("x must be 2D")
    br, bc = block_shape
    if br <= 0 or bc <= 0:
        raise ValueError("block dims must be positive")
    m, n = x.shape
    pm, pn = (br - m % br) % br, (bc - n % bc) % bc
    xp = torch.nn.functional.pad(x, (0, pn, 0, pm))
    blocks = xp.view((m + pm)//br, br, (n + pn)//bc, bc).permute(0, 2, 1, 3)
    scale = torch.clamp(torch.amax(torch.abs(blocks.to(torch.float32)), dim=(-1, -2), keepdim=True) / 127.0, min=EPS)
    qb = torch.round(blocks / scale).clamp(-128, 127).to(torch.int8)
    q = qb.permute(0, 2, 1, 3).contiguous().view(m + pm, n + pn)[:m, :n]
    return q, scale.squeeze(-1).squeeze(-1)


def per_block_dequantization_int8(q: torch.Tensor, scales: torch.Tensor, block_shape: tuple[int, int]) -> torch.Tensor:
    br, bc = block_shape
    m, n = q.shape
    pm, pn = (br - m % br) % br, (bc - n % bc) % bc
    qp = torch.nn.functional.pad(q, (0, pn, 0, pm))
    blocks = qp.view((m + pm)//br, br, (n + pn)//bc, bc).permute(0, 2, 1, 3).to(torch.float32)
    out = blocks * scales[..., None, None].to(torch.float32)
    return out.permute(0, 2, 1, 3).contiguous().view(m + pm, n + pn)[:m, :n]
