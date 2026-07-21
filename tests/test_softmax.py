import torch

from ops.softmax_ops import softmax_safe, softmax_online_rowwise


def test_softmax_online_matches_safe():
    x = torch.randn(4, 16, dtype=torch.float32)
    y_ref = softmax_safe(x, dim=1)
    y_online = softmax_online_rowwise(x)
    assert torch.allclose(y_ref, y_online, atol=1e-5, rtol=1e-5)
