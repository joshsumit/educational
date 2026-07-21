from __future__ import annotations
"""Stage 12.0 — Quantization concepts.

Quantization maps real values into smaller integer or low-precision formats.

Affine quantization:
    q = round(x / scale) + zero_point
    x_hat = (q - zero_point) * scale

Symmetric quantization:
    zero_point = 0
    q = round(x / scale)

Common inference formats:
    W8A8  : int8 weights and int8 activations
    W4A16 : int4 weights and fp16/bf16 activations
    W4A8  : int4 weights and int8 activations
    FP8   : low-precision floating point with per-tensor/per-block scales

Interview point:
    Quantization is not only a math trick. Kernel layout, packing, scale placement, dequant overhead, and memory
    bandwidth determine whether it is actually faster.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantizationScheme:
    name: str
    weight_bits: int
    activation_bits: int | None
    scale_granularity: str
    accumulator: str


def common_schemes() -> list[QuantizationScheme]:
    return [
        QuantizationScheme('W8A8', 8, 8, 'per-tensor/per-channel/per-token', 'int32 or fp32'),
        QuantizationScheme('W4A16', 4, 16, 'per-channel/per-group', 'fp32'),
        QuantizationScheme('FP8', 8, 8, 'per-tensor/per-block', 'fp32'),
    ]


def dequant_formula() -> str:
    return 'x_hat = (q - zero_point) * scale'


def smoke_test() -> None:
    assert common_schemes()[0].name == 'W8A8'
    assert 'scale' in dequant_formula()
