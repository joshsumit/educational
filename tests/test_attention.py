import torch

from ops.matmul_attention_ops import attention_naive, attention_online


def test_attention_online_matches_naive():
    q = torch.randn(12, 16, dtype=torch.float32)
    k = torch.randn(12, 16, dtype=torch.float32)
    v = torch.randn(12, 8, dtype=torch.float32)
    y0 = attention_naive(q, k, v, causal=False)
    y1 = attention_online(q, k, v, causal=False)
    assert torch.allclose(y0, y1, atol=1e-4, rtol=1e-4)
