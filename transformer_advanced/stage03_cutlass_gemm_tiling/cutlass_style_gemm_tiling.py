"""
CUTLASS-style GEMM tiling simulation.

Hierarchy:
    Threadblock/CTA tile: large tile of C, e.g. 128x128x32
    Warp tile: smaller tile inside CTA, e.g. 64x64x32
    MMA tile: hardware instruction tile, e.g. 16x8x16

This implementation computes real matrix multiplication while exposing tile ownership.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class GemmTileConfig:
    cta_m: int = 64
    cta_n: int = 64
    cta_k: int = 32
    warp_m: int = 32
    warp_n: int = 32
    mma_m: int = 16
    mma_n: int = 8
    mma_k: int = 16


def cutlass_style_gemm(a: np.ndarray, b: np.ndarray, cfg: GemmTileConfig = GemmTileConfig()) -> np.ndarray:
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError('inner dimensions must match')
    c = np.zeros((m, n), dtype=np.float32)

    for m0 in range(0, m, cfg.cta_m):
        for n0 in range(0, n, cfg.cta_n):
            cta_m = min(cfg.cta_m, m - m0)
            cta_n = min(cfg.cta_n, n - n0)
            cta_acc = np.zeros((cta_m, cta_n), dtype=np.float32)
            for k0 in range(0, k, cfg.cta_k):
                a_cta = a[m0:m0+cta_m, k0:k0+cfg.cta_k].astype(np.float32)
                b_cta = b[k0:k0+cfg.cta_k, n0:n0+cta_n].astype(np.float32)
                # Warp-level decomposition inside CTA tile.
                for wm in range(0, cta_m, cfg.warp_m):
                    for wn in range(0, cta_n, cfg.warp_n):
                        a_w = a_cta[wm:wm+cfg.warp_m, :]
                        b_w = b_cta[:, wn:wn+cfg.warp_n]
                        cta_acc[wm:wm+a_w.shape[0], wn:wn+b_w.shape[1]] += a_w @ b_w
            c[m0:m0+cta_m, n0:n0+cta_n] = cta_acc
    return c


def tensorcore_utilization_estimate(m: int, n: int, k: int, cfg: GemmTileConfig = GemmTileConfig()) -> dict:
    """Estimate tile counts and edge-tile inefficiency."""
    full_m_tiles = m // cfg.mma_m
    full_n_tiles = n // cfg.mma_n
    full_k_tiles = k // cfg.mma_k
    total_tiles = ((m + cfg.mma_m - 1)//cfg.mma_m) * ((n + cfg.mma_n - 1)//cfg.mma_n) * ((k + cfg.mma_k - 1)//cfg.mma_k)
    full_tiles = full_m_tiles * full_n_tiles * full_k_tiles
    return {'logical_mma_tiles': total_tiles, 'full_mma_tiles': full_tiles, 'edge_tile_fraction': 1.0 - full_tiles / max(total_tiles, 1)}


def smoke_test() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(size=(70,81)).astype(np.float16)
    b = rng.normal(size=(81,37)).astype(np.float16)
    got = cutlass_style_gemm(a,b,GemmTileConfig(cta_m=32,cta_n=32,cta_k=16,warp_m=16,warp_n=16))
    assert np.allclose(got, a.astype(np.float32) @ b.astype(np.float32), atol=1e-2)
    assert tensorcore_utilization_estimate(17,9,17)['logical_mma_tiles'] > 1
