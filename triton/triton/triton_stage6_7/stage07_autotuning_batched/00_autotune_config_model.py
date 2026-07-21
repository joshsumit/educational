from __future__ import annotations
"""Stage 7.0 — Autotune configuration model.

Triton autotuning tries multiple constexpr configurations and chooses the fastest for a shape.

Common matmul tuning parameters:

    BLOCK_M
    BLOCK_N
    BLOCK_K
    GROUP_M
    num_warps
    num_stages

Why tune?
    There is no single best tile for every shape/hardware/dtype.

Tradeoffs:
    - Larger BLOCK_M/N: more reuse, more accumulator registers.
    - Larger BLOCK_K: fewer loop iterations, more input tile footprint.
    - More warps: more parallelism but potentially higher overhead/register pressure.
    - More stages: more pipelining but more resource pressure.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MatmulConfig:
    block_m: int
    block_n: int
    block_k: int
    group_m: int
    num_warps: int
    num_stages: int


def candidate_configs() -> list[MatmulConfig]:
    return [
        MatmulConfig(16, 16, 32, 4, 4, 3),
        MatmulConfig(32, 32, 32, 4, 4, 3),
        MatmulConfig(32, 64, 32, 4, 4, 4),
        MatmulConfig(64, 32, 32, 4, 4, 4),
        MatmulConfig(64, 64, 32, 4, 8, 4),
    ]


def estimate_config_cost(m: int, n: int, k: int, cfg: MatmulConfig) -> float:
    """Simple educational cost model. Lower is better.

    This is intentionally approximate. Real autotuning measures actual runtime.
    """
    tiles_m = math.ceil(m / cfg.block_m)
    tiles_n = math.ceil(n / cfg.block_n)
    tiles_k = math.ceil(k / cfg.block_k)
    total_tile_steps = tiles_m * tiles_n * tiles_k
    reuse_score = (cfg.block_m * cfg.block_n) / max(cfg.block_m + cfg.block_n, 1)
    resource_penalty = 1.0 + 0.05 * max(cfg.num_warps - 4, 0) + 0.03 * max(cfg.num_stages - 3, 0)
    return total_tile_steps * resource_penalty / max(reuse_score, 1.0)


def choose_config(m: int, n: int, k: int, configs: list[MatmulConfig] | None = None) -> MatmulConfig:
    configs = configs or candidate_configs()
    return min(configs, key=lambda c: estimate_config_cost(m, n, k, c))


def smoke_test() -> None:
    cfg = choose_config(256, 256, 256)
    assert cfg in candidate_configs()
    assert estimate_config_cost(256, 256, 256, cfg) > 0
