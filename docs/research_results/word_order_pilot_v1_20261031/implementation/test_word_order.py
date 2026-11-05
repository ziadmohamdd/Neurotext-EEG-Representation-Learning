import numpy as np
from eeg_decoding.evaluation.word_order import fit_projector, pooled, missingness, select_alpha


def test_order_pooling_and_permutation_preserve_mask_and_mean():
    projector = {"center": np.zeros(2), "scale": np.ones(2),
                 "pca_mean": np.zeros(2), "components": np.eye(2)}
    words = np.array([[1, 2], [np.nan, np.nan], [3, 4], [7, 8], [9, 10]], float)
    original = words.copy()
    np.testing.assert_allclose(pooled(words, projector, 1)[:2], [5, 6])
    np.testing.assert_array_equal(words, original)
    a, b = pooled(words, projector, 4), pooled(words, projector, 4, 42)
    np.testing.assert_array_equal(a[-4:], b[-4:])
    np.testing.assert_allclose(pooled(words, projector, 1), pooled(words, projector, 1, 42))
    assert not np.array_equal(a[:8], b[:8])
    np.testing.assert_array_equal(missingness(words)[:4], a[-4:])


def test_empty_and_all_missing_are_retained():
    projector = {"center": np.zeros(2), "scale": np.ones(2),
                 "pca_mean": np.zeros(2), "components": np.eye(2)}
    for words in (np.empty((0, 2)), np.full((3, 2), np.nan)):
        assert np.isfinite(pooled(words, projector)).all()
        np.testing.assert_array_equal(pooled(words, projector), np.zeros(12))
        assert missingness(words)[4] == len(words)


def test_projector_only_depends_on_supplied_training_words():
    words = [np.random.default_rng(i).normal(size=(8, 5)) for i in range(3)]
    fitted = fit_projector(words, components=2, max_words=12)
    other = fit_projector(words, components=2, max_words=12)
    for key in ("center", "scale", "components", "pca_mean"):
        np.testing.assert_array_equal(fitted[key], other[key])
    assert fitted["training_words"] == 24 and fitted["pca_fit_words"] == 12
    assert np.isfinite(pooled(np.full((3, 5), 1e8), fitted)).all()


def test_alpha_selection_uses_recall_and_prefers_stronger_regularization_on_ties():
    rows = [{"alpha": a, "macro_reader_recall@1": score}
            for a, score in [(1, .1), (10, .2), (100, .2)]]
    assert select_alpha(rows) == 100
