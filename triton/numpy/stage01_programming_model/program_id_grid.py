"""Triton programming model basics: grid and program ids.

Triton launches a grid of programs. Each program operates on a block of elements.
Unlike CUDA, you usually reason at block/program granularity rather than individual threads.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

@dataclass(frozen=True)
class Grid1D:
    n_elements: int
    block_size: int
    @property
    def num_programs(self) -> int:
        return math.ceil(self.n_elements / self.block_size)
    def offsets_for_pid(self, pid: int) -> list[int]:
        start = pid * self.block_size
        return list(range(start, min(start+self.block_size, self.n_elements)))

@dataclass(frozen=True)
class Grid2D:
    m: int; n: int; block_m: int; block_n: int
    @property
    def shape(self) -> tuple[int,int]:
        return (math.ceil(self.m/self.block_m), math.ceil(self.n/self.block_n))
    def tile_for_program(self, pid_m: int, pid_n: int) -> tuple[slice,slice]:
        return (slice(pid_m*self.block_m, min((pid_m+1)*self.block_m,self.m)),
                slice(pid_n*self.block_n, min((pid_n+1)*self.block_n,self.n)))

def grouped_matmul_program_order(num_m: int, num_n: int, group_m: int) -> list[tuple[int,int]]:
    """Simulate grouped ordering used to improve L2 reuse in matmul kernels."""
    order=[]
    for g0 in range(0, num_m, group_m):
        g = min(group_m, num_m-g0)
        for n in range(num_n):
            for mi in range(g): order.append((g0+mi,n))
    return order

def smoke_test() -> None:
    g=Grid1D(10,4); assert g.num_programs==3 and g.offsets_for_pid(2)==[8,9]
    g2=Grid2D(9,7,4,3); assert g2.shape==(3,3)
    assert grouped_matmul_program_order(4,2,2)==[(0,0),(1,0),(0,1),(1,1),(2,0),(3,0),(2,1),(3,1)]
