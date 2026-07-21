from __future__ import annotations
import torch
from .stage00_foundations import integer_range
from .stage01_int8_quantization import quantize_affine, dequantize_affine
from .constants import NF4_CODEBOOK, EPS


def int4_quantize_naive(x: torch.Tensor, scale: float | torch.Tensor) -> torch.Tensor:
    return quantize_affine(x, scale, 0, 4, True)


def int4_dequantize_naive(q: torch.Tensor, scale: float | torch.Tensor) -> torch.Tensor:
    return dequantize_affine(q, scale, 0)


def uint4_quantize_naive(x: torch.Tensor, scale: float | torch.Tensor, zero_point: int | torch.Tensor = 8) -> torch.Tensor:
    return quantize_affine(x, scale, zero_point, 4, False)


def uint4_dequantize_naive(q: torch.Tensor, scale: float | torch.Tensor, zero_point: int | torch.Tensor = 8) -> torch.Tensor:
    return dequantize_affine(q, scale, zero_point)


def int2_quantize_naive(x: torch.Tensor, scale: float | torch.Tensor) -> torch.Tensor:
    return quantize_affine(x, scale, 0, 2, True)


def pack_int4(q: torch.Tensor, signed: bool = True) -> torch.Tensor:
    vals = q.to(torch.int16)
    if signed:
        vals = vals + 8
    vals = vals.clamp(0, 15).to(torch.uint8).flatten()
    if vals.numel() % 2:
        vals = torch.cat([vals, torch.zeros(1, dtype=torch.uint8, device=vals.device)])
    return (vals[0::2] & 0x0F) | ((vals[1::2] & 0x0F) << 4)


def unpack_int4(packed: torch.Tensor, original_numel: int, signed: bool = True) -> torch.Tensor:
    p = packed.to(torch.uint8).flatten()
    lo = p & 0x0F
    hi = (p >> 4) & 0x0F
    vals = torch.stack([lo, hi], dim=1).flatten()[:original_numel].to(torch.int16)
    if signed:
        vals = vals - 8
        return vals.to(torch.int8)
    return vals.to(torch.uint8)


def pack_int2(q: torch.Tensor, signed: bool = True) -> torch.Tensor:
    vals = q.to(torch.int16)
    if signed:
        vals = vals + 2
    vals = vals.clamp(0, 3).to(torch.uint8).flatten()
    pad = (-vals.numel()) % 4
    if pad:
        vals = torch.cat([vals, torch.zeros(pad, dtype=torch.uint8, device=vals.device)])
    return vals[0::4] | (vals[1::4] << 2) | (vals[2::4] << 4) | (vals[3::4] << 6)


def unpack_int2(packed: torch.Tensor, original_numel: int, signed: bool = True) -> torch.Tensor:
    p = packed.to(torch.uint8).flatten()
    vals = torch.stack([(p >> shift) & 0x03 for shift in (0, 2, 4, 6)], dim=1).flatten()[:original_numel].to(torch.int16)
    if signed:
        vals = vals - 2
        return vals.to(torch.int8)
    return vals.to(torch.uint8)


def nf4_quantize_reference(x: torch.Tensor, block_size: int | None = None):
    cb = NF4_CODEBOOK.to(device=x.device, dtype=torch.float32)
    flat = x.flatten().to(torch.float32)
    if block_size is None:
        scale = torch.clamp(torch.max(torch.abs(flat)), min=EPS)
        xn = torch.clamp(flat / scale, -1.0, 1.0)
        idx = torch.argmin(torch.abs(xn[:, None] - cb[None, :]), dim=1).to(torch.uint8).view_as(x)
        return idx, scale
    pad = (-flat.numel()) % block_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, device=x.device)])
    blocks = flat.view(-1, block_size)
    scales = torch.clamp(torch.amax(torch.abs(blocks), dim=1), min=EPS)
    xn = torch.clamp(blocks / scales[:, None], -1.0, 1.0)
    idx = torch.argmin(torch.abs(xn[..., None] - cb), dim=-1).to(torch.uint8).flatten()[:x.numel()].view_as(x)
    return idx, scales


def nf4_dequantize_reference(indices: torch.Tensor, scale: torch.Tensor | float, block_size: int | None = None) -> torch.Tensor:
    cb = NF4_CODEBOOK.to(indices.device)
    vals = cb[indices.to(torch.long).flatten()]
    if block_size is None:
        return (vals * torch.as_tensor(scale, device=indices.device)).view_as(indices)
    flat = vals
    pad = (-flat.numel()) % block_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, device=indices.device)])
    out = flat.view(-1, block_size) * torch.as_tensor(scale, device=indices.device).to(torch.float32)[:, None]
    return out.flatten()[:indices.numel()].view_as(indices)
