"""Environment helpers.

This repo can be studied CPU-only. Triton files are present, but optional.
"""
from __future__ import annotations
import importlib.util


def has_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def environment_report() -> dict:
    return {'has_numpy': has_package('numpy'), 'has_triton': has_package('triton'), 'has_torch': has_package('torch')}


def smoke_test() -> None:
    r=environment_report(); assert r['has_numpy'] is True
