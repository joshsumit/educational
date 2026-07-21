import pytest
torch = pytest.importorskip("torch")

from quantization.stage01_int8_quantization import symmetric_int8_quantize_naive, symmetric_int8_dequantize
from quantization.stage04_low_bit_quantization import pack_int4, unpack_int4, nf4_quantize_reference, nf4_dequantize_reference
from quantization.stage07_quantized_gemm import true_int8_gemm_int32_naive, int8_gemm_dequantize
from quantization.stage12_end_to_end_pipeline import QuantizedLinear


def test_int8_roundtrip_shape():
    x = torch.randn(8, 16)
    q, s = symmetric_int8_quantize_naive(x)
    y = symmetric_int8_dequantize(q, s)
    assert q.dtype == torch.int8
    assert y.shape == x.shape


def test_int4_pack_unpack():
    q = torch.tensor([-8, -7, -1, 0, 1, 7], dtype=torch.int8)
    p = pack_int4(q, signed=True)
    u = unpack_int4(p, q.numel(), signed=True)
    assert torch.equal(q, u)


def test_nf4_roundtrip_shape():
    x = torch.randn(33)
    idx, s = nf4_quantize_reference(x, block_size=16)
    y = nf4_dequantize_reference(idx, s, block_size=16)
    assert y.shape == x.shape


def test_gemm_shape():
    a = torch.randint(-10, 10, (4, 8), dtype=torch.int8)
    b = torch.randint(-10, 10, (8, 5), dtype=torch.int8)
    acc = true_int8_gemm_int32_naive(a, b)
    y = int8_gemm_dequantize(acc, 0.1, 0.2)
    assert y.shape == (4, 5)


def test_quantized_linear_reference():
    w = torch.randn(6, 4)
    b = torch.randn(6)
    x = torch.randn(3, 4)
    ql = QuantizedLinear.from_float(w, b)
    y = ql.forward_reference(x)
    assert y.shape == (3, 6)
