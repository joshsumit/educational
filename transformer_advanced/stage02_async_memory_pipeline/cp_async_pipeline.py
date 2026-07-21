"""
`cp.async` pipeline simulation.

Real CUDA idea:
    `cp.async` copies data from global memory to shared memory asynchronously.
    While one tile is being computed, the next tile is being loaded.

This file models the dependency schedule, not real GPU latency.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class PipelineEvent:
    step: int
    action: str
    tile_id: int
    buffer_id: int


def cp_async_schedule(num_tiles: int, stages: int = 2) -> list[PipelineEvent]:
    """
    Create a software-pipeline event list.

    Prologue:
        load initial tiles
    Main loop:
        compute tile i while loading tile i + stages
    Epilogue:
        compute remaining loaded tiles
    """
    events: list[PipelineEvent] = []
    step = 0
    for t in range(min(stages, num_tiles)):
        events.append(PipelineEvent(step, 'cp.async.load.global.to.shared', t, t % stages))
        step += 1
    for t in range(num_tiles):
        next_t = t + stages
        if next_t < num_tiles:
            events.append(PipelineEvent(step, 'cp.async.load.global.to.shared', next_t, next_t % stages))
        events.append(PipelineEvent(step, 'mma.compute.uses.shared', t, t % stages))
        step += 1
    return events


def pipelined_dot(a: np.ndarray, b: np.ndarray, block_k: int = 32, stages: int = 2) -> float:
    """
    Dot product organized as a staged K-loop.

    This is the scalar equivalent of GEMM mainloop pipelining.
    """
    if a.shape != b.shape:
        raise ValueError('a and b must have same shape')
    acc = 0.0
    for k0 in range(0, a.size, block_k):
        # Real kernel would have loaded this block earlier via cp.async.
        a_tile = a[k0:k0+block_k]
        b_tile = b[k0:k0+block_k]
        acc += float(a_tile.astype(np.float32) @ b_tile.astype(np.float32))
    return acc


def smoke_test() -> None:
    sched = cp_async_schedule(5, 2)
    assert any(e.action.startswith('cp.async') for e in sched)
    a = np.arange(100, dtype=np.float16)
    b = np.ones(100, dtype=np.float16)
    assert abs(pipelined_dot(a, b, 16) - float(a.astype(np.float32) @ b.astype(np.float32))) < 1e-4
