import torch

from ops.quantization_ops import (
    dequantize_per_tensor_int8,
    fixed_point_matmul_int8,
    quantize_per_tensor_int8,
)


def test_fixed_point_matmul_shapes_and_types():
    a = torch.randn(6, 10, dtype=torch.float32)
    b = torch.randn(10, 5, dtype=torch.float32)

    a_scale = 0.05
    b_scale = 0.04
    out_scale = 0.07

    a_q = quantize_per_tensor_int8(a, a_scale, 0)
    b_q = quantize_per_tensor_int8(b, b_scale, 0)

    out_q, acc = fixed_point_matmul_int8(
        a_q=a_q,
        b_q=b_q,
        a_scale=a_scale,
        b_scale=b_scale,
        out_scale=out_scale,
    )

    assert out_q.shape == (6, 5)
    assert acc.shape == (6, 5)
    assert out_q.dtype == torch.int8
    assert acc.dtype == torch.int32


def test_quantize_dequantize_roundtrip_reasonable_error():
    x = torch.randn(32, dtype=torch.float32)
    scale = 0.1
    q = quantize_per_tensor_int8(x, scale, 0)
    x_hat = dequantize_per_tensor_int8(q, scale, 0)
    mae = torch.mean(torch.abs(x - x_hat)).item()
    assert mae < 0.15
