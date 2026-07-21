from __future__ import annotations
import torch
from .constants import EPS


def minmax_calibration(samples: torch.Tensor | list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    xs = samples if isinstance(samples, (list, tuple)) else [samples]
    mn = torch.min(torch.stack([torch.min(t.to(torch.float32)) for t in xs]))
    mx = torch.max(torch.stack([torch.max(t.to(torch.float32)) for t in xs]))
    return mn, mx


def percentile_calibration(x: torch.Tensor, percentile: float = 99.99, symmetric: bool = True):
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    xf = x.flatten().to(torch.float32)
    if symmetric:
        a = torch.quantile(torch.abs(xf), percentile / 100.0)
        return -a, a
    lo = torch.quantile(xf, (100.0 - percentile) / 200.0)
    hi = torch.quantile(xf, 1.0 - (100.0 - percentile) / 200.0)
    return lo, hi


def _fake_quant_mse(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, num_bits: int = 8):
    qmin, qmax = -128, 127
    scale = torch.clamp((hi - lo) / float(qmax - qmin), min=EPS)
    zp = torch.round(qmin - lo / scale).clamp(qmin, qmax)
    q = torch.round(x / scale + zp).clamp(qmin, qmax)
    dq = (q - zp) * scale
    return torch.mean((x - dq) ** 2)


def mse_calibration(x: torch.Tensor, num_bits: int = 8, symmetric: bool = True, grid_size: int = 100):
    xf = x.flatten().to(torch.float32)
    if symmetric:
        max_abs = torch.max(torch.abs(xf))
        best_a, best_mse = max_abs, torch.tensor(float('inf'), device=x.device)
        for shrink in torch.linspace(0.5, 1.0, grid_size, device=x.device):
            a = max_abs * shrink
            mse = _fake_quant_mse(torch.clamp(xf, -a, a), -a, a, num_bits)
            if mse < best_mse:
                best_a, best_mse = a, mse
        return -best_a, best_a
    mn, mx = torch.min(xf), torch.max(xf)
    best = (mn, mx, torch.tensor(float('inf'), device=x.device))
    for shrink in torch.linspace(0.5, 1.0, grid_size, device=x.device):
        lo, hi = mn * shrink, mx * shrink
        mse = _fake_quant_mse(torch.clamp(xf, lo, hi), lo, hi, num_bits)
        if mse < best[2]:
            best = (lo, hi, mse)
    return best[0], best[1]


def histogram_calibration(x: torch.Tensor, bins: int = 2048, symmetric: bool = True):
    xf = x.flatten().to(torch.float32)
    if symmetric:
        a = torch.max(torch.abs(xf))
        hist = torch.histc(torch.clamp(xf, -a, a), bins=bins, min=float(-a), max=float(a))
        edges = torch.linspace(-float(a), float(a), steps=bins + 1, device=x.device)
    else:
        mn, mx = torch.min(xf), torch.max(xf)
        hist = torch.histc(xf, bins=bins, min=float(mn), max=float(mx))
        edges = torch.linspace(float(mn), float(mx), steps=bins + 1, device=x.device)
    return hist, edges


def kl_divergence_calibration(x: torch.Tensor, num_bits: int = 8, bins: int = 2048, quant_bins: int = 255):
    hist, edges = histogram_calibration(x, bins=bins, symmetric=True)
    hist = hist.to(torch.float64)
    best_thr, best_kl = edges[-1].abs(), float('inf')
    center = bins // 2
    for half in range(max(quant_bins//2, 1), center):
        lo, hi = center - half, center + half
        p = hist[lo:hi].clone()
        p[0] += hist[:lo].sum(); p[-1] += hist[hi:].sum()
        if p.sum() <= 0:
            continue
        p = p / p.sum()
        bucket = max(1, p.numel() // quant_bins)
        q = torch.zeros_like(p)
        for i in range(0, p.numel(), bucket):
            sl = slice(i, min(i+bucket, p.numel()))
            mass = p[sl].sum()
            nonzero = torch.clamp((p[sl] > 0).sum(), min=1)
            q[sl] = mass / nonzero
        kl = torch.sum(p * torch.log(torch.clamp(p, min=EPS) / torch.clamp(q, min=EPS))).item()
        if kl < best_kl:
            best_kl = kl; best_thr = edges[hi].abs()
    return -best_thr, best_thr
