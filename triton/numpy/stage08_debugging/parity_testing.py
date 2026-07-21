"""Triton debugging helpers: parity, tolerances, and worst-case diagnostics."""
from __future__ import annotations
import numpy as np

def compare_tensors(actual: np.ndarray, expected: np.ndarray, rtol:float=1e-4, atol:float=1e-4) -> dict:
    diff=np.abs(actual-expected); denom=np.maximum(np.abs(expected), atol)
    rel=diff/denom; idx=np.unravel_index(int(np.argmax(diff)), diff.shape)
    return {'ok': bool(np.allclose(actual,expected,rtol=rtol,atol=atol)), 'max_abs': float(diff[idx]), 'max_rel': float(np.max(rel)), 'worst_index': tuple(int(i) for i in idx), 'actual': float(actual[idx]), 'expected': float(expected[idx])}

def smoke_test() -> None:
    a=np.array([1.0,2.0]); b=np.array([1.0,2.001])
    r=compare_tensors(a,b,atol=1e-2); assert r['ok'] is True and r['worst_index']==(1,)
