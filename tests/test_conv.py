import torch

from ops.tier1_core_nn_ops import conv2d_naive, conv2d_optimized


def test_conv2d_naive_matches_optimized():
    x = torch.randn(2, 3, 8, 8, dtype=torch.float32)
    w = torch.randn(4, 3, 3, 3, dtype=torch.float32)
    b = torch.randn(4, dtype=torch.float32)
    y0 = conv2d_naive(x, w, bias=b, stride=(1, 1), padding=(1, 1))
    y1 = conv2d_optimized(x, w, bias=b, stride=(1, 1), padding=(1, 1))
    assert torch.allclose(y0, y1, atol=1e-4, rtol=1e-4)
