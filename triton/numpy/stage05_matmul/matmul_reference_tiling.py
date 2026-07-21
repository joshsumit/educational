"""Matmul progression: naive -> blocked -> grouped program order."""
from __future__ import annotations
import numpy as np
from stage01_programming_model.program_id_grid import grouped_matmul_program_order

def matmul_naive(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    m,k=a.shape; k2,n=b.shape
    if k!=k2: raise ValueError('shape mismatch')
    c=np.zeros((m,n),dtype=np.float32)
    for i in range(m):
        for j in range(n):
            acc=0.0
            for kk in range(k): acc += float(a[i,kk])*float(b[kk,j])
            c[i,j]=acc
    return c

def matmul_blocked(a: np.ndarray, b: np.ndarray, block_m:int=16, block_n:int=16, block_k:int=32, group_m:int=4) -> np.ndarray:
    m,k=a.shape; _,n=b.shape; c=np.zeros((m,n),dtype=np.float32)
    num_m=(m+block_m-1)//block_m; num_n=(n+block_n-1)//block_n
    for pid_m,pid_n in grouped_matmul_program_order(num_m,num_n,group_m):
        ms=slice(pid_m*block_m,min((pid_m+1)*block_m,m)); ns=slice(pid_n*block_n,min((pid_n+1)*block_n,n))
        acc=np.zeros((ms.stop-ms.start, ns.stop-ns.start), dtype=np.float32)
        for k0 in range(0,k,block_k):
            acc += a[ms,k0:k0+block_k].astype(np.float32) @ b[k0:k0+block_k,ns].astype(np.float32)
        c[ms,ns]=acc
    return c

def smoke_test() -> None:
    rng=np.random.default_rng(2); a=rng.normal(size=(17,19)).astype(np.float32); b=rng.normal(size=(19,11)).astype(np.float32)
    assert np.allclose(matmul_naive(a,b), a@b, atol=1e-5)
    assert np.allclose(matmul_blocked(a,b,5,4,7,2), a@b, atol=1e-5)
