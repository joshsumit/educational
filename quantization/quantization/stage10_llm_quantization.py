from __future__ import annotations
import torch
from .stage02_granularity import per_group_quantization_int8, per_group_dequantization_int8
from .stage04_low_bit_quantization import int4_quantize_naive, int4_dequantize_naive, nf4_quantize_reference, nf4_dequantize_reference
from .constants import EPS


def smoothquant(x_samples: torch.Tensor, weight: torch.Tensor, alpha: float = 0.5):
    # x_samples: [tokens, in_features], weight: [out_features, in_features]
    act_scale = torch.clamp(torch.amax(torch.abs(x_samples.to(torch.float32)), dim=0), min=EPS)
    weight_scale = torch.clamp(torch.amax(torch.abs(weight.to(torch.float32)), dim=0), min=EPS)
    s = torch.pow(act_scale, alpha) / torch.pow(weight_scale, 1.0 - alpha)
    x_smooth = x_samples / s
    w_smooth = weight * s
    return x_smooth, w_smooth, s


def gptq_hessian_approx(act_samples: torch.Tensor, damping: float = 0.01) -> torch.Tensor:
    x = act_samples.to(torch.float32)
    h = (x.t() @ x) / max(1, x.shape[0])
    diag = torch.mean(torch.diag(h)) * damping
    return h + torch.eye(h.shape[0], device=h.device, dtype=h.dtype) * diag


def gptq_block_solve(weight: torch.Tensor, hessian: torch.Tensor, bits: int = 4, group_size: int = 128):
    # Educational GPTQ-like sequential error compensation over columns.
    w = weight.to(torch.float32).clone()
    h_inv = torch.linalg.pinv(hessian.to(torch.float32))
    q = torch.empty_like(w, dtype=torch.int8)
    scales = []
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    for start in range(0, w.shape[1], group_size):
        end = min(start + group_size, w.shape[1])
        block = w[:, start:end]
        scale = torch.clamp(torch.amax(torch.abs(block), dim=1, keepdim=True) / qmax, min=EPS)
        qb = torch.round(block / scale).clamp(qmin, qmax)
        dq = qb * scale
        err = block - dq
        q[:, start:end] = qb.to(torch.int8)
        scales.append(scale.squeeze(1))
        if end < w.shape[1]:
            correction = err @ h_inv[start:end, end:]
            w[:, end:] -= correction
    return q, torch.stack(scales, dim=1)


def gptq_style_quantization_reference(w: torch.Tensor, group_size: int = 128):
    return per_group_quantization_int8(w, group_size=group_size, axis=-1)


def awq_style_quantization_reference(w: torch.Tensor, act_samples: torch.Tensor, group_size: int = 128, protect_ratio: float = 0.01):
    importance = torch.mean(torch.abs(act_samples.to(torch.float32)), dim=0)
    k = max(1, int(importance.numel() * protect_ratio))
    protected = torch.zeros_like(importance, dtype=torch.bool)
    protected[torch.topk(importance, k).indices] = True
    scale_boost = torch.ones_like(importance)
    scale_boost[protected] = 0.5
    w_scaled = w * scale_boost.view(1, -1)
    q, scales = per_group_quantization_int8(w_scaled, group_size, axis=-1)
    return q, scales, protected


def hqq_reference(w: torch.Tensor, bits: int = 4, iters: int = 3):
    qmax = (1 << (bits - 1)) - 1; qmin = -(1 << (bits - 1))
    scale = torch.clamp(torch.amax(torch.abs(w), dim=1, keepdim=True) / qmax, min=EPS)
    zero = torch.zeros_like(scale)
    q = torch.round(w / scale).clamp(qmin, qmax)
    for _ in range(iters):
        dq = q * scale + zero
        residual = w - dq
        zero = zero + residual.mean(dim=1, keepdim=True)
        scale = torch.clamp(((w - zero) * q).sum(dim=1, keepdim=True) / torch.clamp((q*q).sum(dim=1, keepdim=True), min=EPS), min=EPS)
        q = torch.round((w - zero) / scale).clamp(qmin, qmax)
    return q.to(torch.int8), scale.squeeze(1), zero.squeeze(1)


def gguf_q4_0_quantization(w: torch.Tensor, block_size: int = 32):
    flat = w.flatten().to(torch.float32)
    pad = (-flat.numel()) % block_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, device=w.device)])
    blocks = flat.view(-1, block_size)
    scale = torch.clamp(torch.amax(torch.abs(blocks), dim=1, keepdim=True) / 7.0, min=EPS)
    q = torch.round(blocks / scale).clamp(-8, 7).to(torch.int8)
    return q.flatten()[:w.numel()].view_as(w), scale.squeeze(1)


def mxfp4_quantization(x: torch.Tensor, block_size: int = 32):
    # Microscaling-style shared power-of-two scale with signed E2M1-like 4-bit values approximated by uniform int4.
    flat = x.flatten().to(torch.float32)
    pad = (-flat.numel()) % block_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, device=x.device)])
    b = flat.view(-1, block_size)
    amax = torch.clamp(torch.amax(torch.abs(b), dim=1, keepdim=True), min=EPS)
    scale = torch.pow(2.0, torch.floor(torch.log2(amax / 6.0)))
    q = torch.round(b / scale).clamp(-8, 7).to(torch.int8)
    return q.flatten()[:x.numel()].view_as(x), scale.squeeze(1)
