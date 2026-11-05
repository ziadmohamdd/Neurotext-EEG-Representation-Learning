import numpy as np
import pytest
from eeg_decoding.evaluation.relative_trt import relative_mean, score_ranks, choose


def test_relative_band_mean_and_word_mean_are_exact():
    x = np.arange(2 * 840, dtype=float).reshape(2, 840)
    z = np.log1p(x).reshape(2, 8, 105)
    expected = (z - z.mean(2, keepdims=True)).mean(0).ravel()
    np.testing.assert_allclose(relative_mean(x), expected)
    np.testing.assert_allclose(relative_mean(x).reshape(8, 105).mean(1), 0, atol=1e-14)
    np.testing.assert_allclose(relative_mean(np.r_[x, np.full((1, 840), np.nan)]), expected)


def test_relative_missingness_and_invalid_input():
    assert np.isnan(relative_mean(np.full((3, 840), np.nan))).all()
    assert np.isnan(relative_mean(np.empty((0, 840)))).all()
    np.testing.assert_array_equal(relative_mean(np.full((1, 840), -1.)), np.zeros(840))
    x = np.zeros((1, 840)); x[0, 0] = np.nan
    with pytest.raises(ValueError):
        relative_mean(x)


def test_classifier_ranking_uses_average_ties_and_raw_scores():
    np.testing.assert_array_equal(score_ranks([[3, 2, 1], [1, 1, 1], [0, 4, 2]], [0, 1, 2]), [1, 2, 2])


def test_selection_prefers_zero_then_larger_alpha_for_ties():
    rows = [{"weight": w, "alpha": a, "macro_recall@1": .2}
            for w, a in [(1, 1), (.1, 100), (0, 1), (0, 10)]]
    assert choose(rows) == rows[-1]
