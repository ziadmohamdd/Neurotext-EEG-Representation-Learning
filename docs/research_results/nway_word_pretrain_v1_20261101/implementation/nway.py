"""Exact expected strict top-1 accuracy under uniform N-way distractor draws."""
from math import comb
import numpy as np


def expected_top1(strictly_below, pool_size, n):
    """Choose N-1 distractors without replacement; any top-score tie is a miss.

    The probability is C(b, N-1)/C(M-1, N-1), where b is the number
    of candidates strictly below the truth. This matches average-tied-rank<=1.
    """
    b = np.asarray(strictly_below)
    if pool_size < 2 or not 2 <= n <= pool_size:
        raise ValueError("Require 2 <= N <= pool size")
    if np.any(b < 0) or np.any(b >= pool_size) or not np.all(b == b.astype(int)):
        raise ValueError("Invalid number of candidates strictly below truth")
    denominator = comb(pool_size - 1, n - 1)
    table = np.array([comb(i, n - 1) / denominator if i >= n - 1 else 0.
                      for i in range(pool_size)])
    return table[b.astype(int)]


def score_counts(scores, target_indices):
    scores = np.asarray(scores)
    indices = np.asarray(target_indices, dtype=int)
    if scores.ndim != 2 or len(scores) != len(indices) or not np.isfinite(scores).all():
        raise ValueError("Invalid candidate scores")
    truth = scores[np.arange(len(scores)), indices, None]
    below = (scores < truth).sum(1)
    ties = (scores == truth).sum(1)
    rank = 1 + (scores > truth).sum(1) + .5 * (ties - 1)
    return below, ties, rank
