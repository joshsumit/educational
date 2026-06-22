import numpy as np
from tensor_utils import print_tensor, print_matrix_multiply, subsection, print_tensor_stats


def softmax(x, axis=-1):
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def build_causal_mask(query_tokens: int, context_tokens: int):
    """
    Build lower-triangular mask for prefill.

    Output shape: [query_tokens, context_tokens]
    """
    return np.tril(np.ones((query_tokens, context_tokens), dtype=np.float32))


def scaled_dot_product_attention(Q, K, V, use_causal_mask: bool, phase_name: str):
    """
    Q: [B, H, S, Dk]
    K: [B, H, T, Dk]
    V: [B, H, T, Dk]

    Returns:
        context: [B, H, S, Dk]
        scores: [B, H, S, T]
        weights: [B, H, S, T]
    """
    B, H, S, Dk = Q.shape
    T = K.shape[2]

    subsection(f"ATTENTION START: {phase_name}")
    print_tensor("Q entering attention", Q, "Query tensor for all heads", dim_names=["B", "H", "S", "Dk"])
    print_tensor("K entering attention", K, "Key tensor used as context", dim_names=["B", "H", "T", "Dk"])
    print_tensor("V entering attention", V, "Value tensor used as context", dim_names=["B", "H", "T", "Dk"])

    # Before this matmul we need K^T over the last two dimensions.
    K_t = K.transpose(0, 1, 3, 2)
    print_tensor("K transposed for QK^T", K_t, "K rearranged so last dims become [Dk, T]", dim_names=["B", "H", "Dk", "T"])

    print_matrix_multiply(
        "MatMul for raw attention scores",
        left_shape=Q.shape,
        right_shape=K_t.shape,
        out_shape=(B, H, S, T),
        explanation="Each query vector compares itself against every key in the current context",
    )

    scores = np.matmul(Q, K_t) / np.sqrt(Dk)
    print_tensor("Raw attention scores", scores, "Similarity scores before masking and softmax", dim_names=["B", "H", "S", "T"])
    print_tensor_stats("Raw attention scores", scores)

    if use_causal_mask:
        mask = build_causal_mask(query_tokens=S, context_tokens=T)
        print_tensor("Causal mask", mask, "1 means visible token, 0 means future token blocked", dim_names=["S", "T"])
        scores = scores + np.where(mask == 0, -1e9, 0.0)[None, None, :, :]
        print_tensor("Raw attention scores after causal mask", scores, "Future-token positions are pushed to large negative values", dim_names=["B", "H", "S", "T"])
    else:
        print("\nNo causal mask applied in this call. This is expected during single-token decode because there are no future query positions inside this step.")

    weights = softmax(scores, axis=-1)
    print_tensor("Attention weights after softmax", weights, "Probability distribution over context tokens", dim_names=["B", "H", "S", "T"])

    print_matrix_multiply(
        "MatMul for weighted value aggregation",
        left_shape=weights.shape,
        right_shape=V.shape,
        out_shape=(B, H, S, Dk),
        explanation="Attention weights are used to compute a weighted sum of V for each head",
    )
    context = np.matmul(weights, V)
    print_tensor("Context tensor", context, "Per-head attended output before head merge", dim_names=["B", "H", "S", "Dk"])

    subsection(f"ATTENTION END: {phase_name}")
    return context, scores, weights
