from itertools import combinations
import numpy as np
import pytest
from eeg_decoding.evaluation.nway import expected_top1, score_counts


def test_exact_expectation_matches_exhaustive_shared_draws_with_ties():
    # The same distractor subsets are used for both score vectors.
    scores = np.array([[5., 4., 3., 2., 1.], [3., 4., 3., 2., 1.]])
    truth = np.array([0, 0])
    below, ties, rank = score_counts(scores, truth)
    np.testing.assert_array_equal(below, [4, 2])
    np.testing.assert_array_equal(ties, [1, 2])
    np.testing.assert_array_equal(rank, [1, 2.5])
    for n in (2, 3, 4, 5):
        draws = list(combinations(range(1, 5), n - 1))
        actual = [np.mean([bool(np.all(row[0] > row[list(draw)])) for draw in draws]) for row in scores]
        np.testing.assert_allclose(expected_top1(below, 5, n), actual)


def test_uniform_random_rank_average_equals_chance_and_full_pool_matches_top1():
    below = np.arange(12)
    for n in (2, 3, 6, 12):
        assert expected_top1(below, 12, n).mean() == pytest.approx(1 / n)
    np.testing.assert_array_equal(expected_top1(below, 12, 12), below == 11)


def test_invalid_nway_request_is_rejected():
    for n in (1, 13):
        with pytest.raises(ValueError):
            expected_top1([1], 12, n)
    with pytest.raises(ValueError):
        expected_top1([12], 12, 2)
