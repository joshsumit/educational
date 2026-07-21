"""Stage 0.3 — Optional imports and environment report.

The repository must remain useful on a CPU-only laptop. Therefore, importing these modules should never
fail just because Triton, Torch, CUDA, or ROCm is missing.

Stage 0 and Stage 1 smoke tests require only NumPy/Python. Later kernel stages will include real
@triton.jit kernels guarded behind optional imports.
"""
from __future__ import annotations

import importlib.util
import platform
import sys
from typing import Any


def has_package(name: str) -> bool:
    """Return True if a Python package can be imported."""
    return importlib.util.find_spec(name) is not None


def _safe_version(name: str) -> str | None:
    """Best-effort package version lookup without importing heavy GPU packages unnecessarily."""
    try:
        import importlib.metadata as metadata
        return metadata.version(name)
    except Exception:
        return None


def environment_report() -> dict[str, Any]:
    """Return a CPU-safe environment report.

    The report intentionally avoids forcing `torch.cuda` initialization during normal smoke tests.
    """
    return {
        'python_version': sys.version.split()[0],
        'platform': platform.platform(),
        'has_numpy': has_package('numpy'),
        'numpy_version': _safe_version('numpy'),
        'has_torch': has_package('torch'),
        'torch_version': _safe_version('torch'),
        'has_triton': has_package('triton'),
        'triton_version': _safe_version('triton'),
    }


def print_environment() -> None:
    """Print a readable environment report."""
    for k, v in environment_report().items():
        print(f'{k}: {v}')


def smoke_test() -> None:
    report = environment_report()
    assert report['has_numpy'] is True
    assert isinstance(report['python_version'], str) and report['python_version']


if __name__ == '__main__':
    print_environment()
