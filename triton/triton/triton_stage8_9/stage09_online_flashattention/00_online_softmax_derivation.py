from __future__ import annotations
"""Stage 9.0 — Online softmax derivation.

Normal softmax for one row:

    p_i = exp(x_i - m) / l
    m = max_i x_i
    l = sum_i exp(x_i - m)

If the row arrives in blocks, we can update running statistics.

Given old stats:

    m_old
    l_old

For a new block x_block:

    m_block = max(x_block)
    m_new = max(m_old, m_block)

Old denominator must be rescaled:

    l_old_rescaled = l_old * exp(m_old - m_new)

New block denominator:

    l_block_rescaled = sum(exp(x_block - m_new))

Updated denominator:

    l_new = l_old_rescaled + l_block_rescaled

For attention output, also keep a running numerator accumulator:

    acc = sum_j exp(score_j - m) * V_j

When m changes, acc must be rescaled exactly like l.
"""

import math


def update_online_stats(m_old: float, l_old: float, m_block: float, l_block_at_m_new: float) -> tuple[float, float, float]:
    """Update online softmax stats.

    Args:
        m_old: previous running max, or -inf initially
        l_old: previous denominator under m_old
        m_block: max of the incoming block
        l_block_at_m_new: sum(exp(block - m_new)); caller computes after m_new is known in real algorithm

    Returns:
        m_new, l_new_without_block_bug_note, old_scale

    This function is mainly pedagogical; full block computation appears in the next file.
    """
    m_new = max(m_old, m_block)
    old_scale = 0.0 if m_old == -math.inf else math.exp(m_old - m_new)
    l_new = l_old * old_scale + l_block_at_m_new
    return m_new, l_new, old_scale


def smoke_test() -> None:
    m_new, l_new, old_scale = update_online_stats(-math.inf, 0.0, 2.0, 1.5)
    assert m_new == 2.0
    assert old_scale == 0.0
    assert l_new == 1.5
