"""Mini CPU-backed kernel library mirroring the Triton learning path."""
from __future__ import annotations
import numpy as np
from stage03_elementwise.vector_add import vector_add_blocked
from stage04_reductions.softmax_layernorm import row_softmax, layernorm
from stage05_matmul.matmul_reference_tiling import matmul_blocked
from stage06_fusions.fused_bias_gelu import fused_bias_gelu

class MiniKernelLibrary:
    def add(self,a,b): return vector_add_blocked(a,b)
    def matmul(self,a,b): return matmul_blocked(a,b)
    def softmax(self,x): return row_softmax(x)
    def layernorm(self,x,g,b): return layernorm(x,g,b)
    def bias_gelu(self,x,bias): return fused_bias_gelu(x,bias)

def smoke_test() -> None:
    rng=np.random.default_rng(6); lib=MiniKernelLibrary()
    a=rng.normal(size=(8,9)).astype(np.float32); b=rng.normal(size=(9,7)).astype(np.float32)
    assert np.allclose(lib.matmul(a,b), a@b, atol=1e-5)
    x=rng.normal(size=(3,7)).astype(np.float32); assert lib.softmax(x).shape==x.shape
