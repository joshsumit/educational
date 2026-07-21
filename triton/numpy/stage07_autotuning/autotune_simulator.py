"""Autotuning simulator.

Triton autotuning tries multiple constexpr configurations such as BLOCK_M/BLOCK_N/BLOCK_K/num_warps.
This file models config search with a simple analytical cost.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class MatmulConfig:
    block_m:int; block_n:int; block_k:int; num_warps:int

def estimate_config_cost(m:int,n:int,k:int,cfg:MatmulConfig) -> float:
    tiles=math.ceil(m/cfg.block_m)*math.ceil(n/cfg.block_n)*math.ceil(k/cfg.block_k)
    reuse_bonus=(cfg.block_m*cfg.block_n)/max(cfg.block_m+cfg.block_n,1)
    occupancy_penalty=1.0+0.03*max(cfg.num_warps-4,0)
    return tiles*occupancy_penalty/max(reuse_bonus,1.0)

def choose_best_config(m:int,n:int,k:int, configs:list[MatmulConfig]) -> MatmulConfig:
    return min(configs, key=lambda c: estimate_config_cost(m,n,k,c))

def smoke_test() -> None:
    configs=[MatmulConfig(16,16,32,4),MatmulConfig(32,64,32,4),MatmulConfig(64,64,32,8)]
    assert choose_best_config(256,256,256,configs) in configs
