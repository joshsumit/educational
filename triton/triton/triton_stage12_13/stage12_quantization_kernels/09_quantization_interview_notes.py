from __future__ import annotations
"""Stage 12.9 — Quantization interview notes."""


def int8_matmul_answer() -> str:
    return 'INT8 matmul usually multiplies int8 operands, accumulates into int32, then applies scales to recover fp output.'


def weight_only_answer() -> str:
    return 'Weight-only quantization stores weights in int4/int8 but keeps activations higher precision; the kernel unpacks/dequantizes weights while computing.'


def kv_quant_answer() -> str:
    return 'KV cache quantization reduces inference memory and decode bandwidth, but scale granularity and attention accuracy must be managed carefully.'


def smoke_test() -> None:
    assert 'int32' in int8_matmul_answer()
    assert 'unpacks' in weight_only_answer()
    assert 'decode bandwidth' in kv_quant_answer()
