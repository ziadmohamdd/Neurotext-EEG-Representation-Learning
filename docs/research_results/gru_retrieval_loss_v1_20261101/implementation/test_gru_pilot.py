import numpy as np
import pytest
import torch
from eeg_decoding.evaluation.gru_pilot import SequenceGRU, encode, fit_scaler, reader_folds
from eeg_decoding.evaluation.spectral_pilot import fit_word_moments, matched_gaussian
from eeg_decoding.evaluation.controls import shuffled_keys


def fixture():
    w = np.random.default_rng(3).uniform(0, 4, (5, 840))
    w[1] = np.nan
    return w


def test_missing_positions_and_all_missing_retained():
    w = fixture(); scaler = fit_scaler([w])
    x = encode(w, scaler)
    assert x.shape == (5, 841) and not x[1].any()
    assert x[:, -1].tolist() == [1, 0, 1, 1, 1]
    absent = encode(np.full((5, 840), np.nan), scaler)
    assert absent.shape == x.shape and not absent.any()
    assert np.isfinite(SequenceGRU()(torch.tensor(absent)[None], torch.tensor([5])).detach().numpy()).all()


def test_padding_invariance():
    torch.manual_seed(42); m = SequenceGRU().eval()
    x = torch.randn(2, 5, 841); lengths = torch.tensor([3, 5])
    before = m(x, lengths)
    x[0, 3:] = 1e6
    tail = torch.cat([x, torch.randn(2, 7, 841) * 1e6], 1)
    torch.testing.assert_close(before, m(tail, lengths), atol=1e-6, rtol=1e-6)


def test_training_only_scaler_and_gaussian_moments():
    w = fixture(); q = w.copy(); scaler = fit_scaler([w]); moments = fit_word_moments([w])
    before = {k: v.copy() for k, v in scaler.items()}
    q[np.isfinite(q)] = 1e8
    encode(q, scaler)
    for k in scaler: np.testing.assert_array_equal(scaler[k], before[k])
    np.testing.assert_allclose(moments['mean'], np.nanmean(w, 0))
    noise = matched_gaussian(q, moments, 42)
    np.testing.assert_array_equal(np.isfinite(q), np.isfinite(noise))


def test_control_matching():
    w = fixture(); scaler = fit_scaler([w]); x = encode(w, scaler)
    order = encode(w, scaler, 'word_order_shuffled')
    np.testing.assert_array_equal(x[:, -1], order[:, -1])
    np.testing.assert_allclose(x[:, :840].mean(0), order[:, :840].mean(0), atol=1e-6)
    missing = encode(w, scaler, 'missingness_only')
    assert not missing[:, :840].any()
    np.testing.assert_array_equal(x[:, -1], missing[:, -1])
    keys = ['a','b','a','c']; shuffled = shuffled_keys(keys)
    assert shuffled[0] == shuffled[2] and all(a != b for a,b in zip(keys,shuffled))


def test_reader_separation():
    subjects = np.repeat(['ZDM','ZDN','ZGW','ZJM','ZJN','ZKH','ZMG','ZPH'], 2)
    for s,a,b in reader_folds(subjects):
        assert not set(a) & set(b) and set(subjects[b]) == {s} and s not in subjects[a]
    with pytest.raises(ValueError): reader_folds(np.r_[subjects, 'ZJS'])
