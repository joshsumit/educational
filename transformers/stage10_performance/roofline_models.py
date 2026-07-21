"""
Roofline and bandwidth models for Transformer kernels.

Arithmetic intensity:
    FLOPs / bytes moved

If intensity is low, kernel is memory-bound.
If intensity is high, kernel may be compute-bound.
Decode attention is often memory-bound because each step streams the whole K/V history for one query.
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class KernelCost:
    flops: float
    bytes: float

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / max(self.bytes, 1.0)


def qk_gemm_cost(batch: int, heads: int, tq: int, tk: int, dh: int, bytes_per_elem: int = 2) -> KernelCost:
    flops = 2 * batch * heads * tq * tk * dh
    # Simplified: read Q and K once, write S once.
    bytes_moved = bytes_per_elem * (batch*heads*tq*dh + batch*heads*tk*dh + batch*heads*tq*tk)
    return KernelCost(flops, bytes_moved)


def decode_attention_cost(seq: int, hq: int, hkv: int, dh: int, bytes_per_elem: int = 2) -> KernelCost:
    flops = 4 * hq * seq * dh  # QK dot and PV weighted sum, rough count
    bytes_moved = bytes_per_elem * 2 * hkv * seq * dh
    return KernelCost(flops, bytes_moved)


def classify(cost: KernelCost, peak_flops: float, peak_bandwidth: float) -> str:
    attainable_by_bw = cost.arithmetic_intensity * peak_bandwidth
    return 'memory_bound' if attainable_by_bw < peak_flops else 'compute_bound'


def smoke_test() -> None:
    c = decode_attention_cost(1024, 32, 8, 128)
    assert c.flops > 0 and c.bytes > 0
    assert classify(c, 100e12, 2e12) in {'memory_bound','compute_bound'}
