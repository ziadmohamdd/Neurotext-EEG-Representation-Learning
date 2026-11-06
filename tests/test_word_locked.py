"""SYNTHETIC fixtures only. These test the word-locked software, not EEG performance."""
import importlib.util
import os
from pathlib import Path
import numpy as np
import pytest
from eeg_decoding.data.dataset import Record
from eeg_decoding.data.word_rows import band_vector, waveform_bins
from eeg_decoding.evaluation import word_locked as locked
from eeg_decoding.evaluation.averaged_eeg import Layout
from eeg_decoding.evaluation.word_locked import (FIRST, SECOND, THIRD, GramSolver, Table, WordData, WordScored, half_readers, noise_options, read_outcome,
                                                 run_arm, run_test, scale_rows, seen_type_share, sentence_means, shuffled_table)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('word_locked_v6', ROOT/'scripts/run_word_locked.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
# The frozen plan's check used 20 seeds; the everyday suite keeps 5 for speed.
SEEDS = range(int(os.environ.get('WORD_LOCKED_SEEDS', 5)))
PLAN = {'seed': 42, 'model': {'alphas': [1, 100, 10000]}, 'arms_shuffles': 2, 'decision_rule': {'minimum_gap_points': 2.},
        'sentence_prediction': {'reader_counts_for_curve': [1, 2, 'all'], 'subset_draws_count': 2}}
SENTENCES, READERS, DIMENSION, TYPES = 120, 6, 16, 400


def layout():
    records = []
    for release, offset in (('zuco1', 0), ('zuco2', 1000)):
        for reader in range(READERS):
            for i in range(SENTENCES):
                records.append(Record(release, f'{release[-1]}R{reader}', '1', 'nr', str(i), f's{offset + i} x', f'{release}_{reader}_{i}.npz', ['C3'], 200.,
                                      provenance={'synthetic': True}))
    zone = {'zuco1': set(range(100, 120)), 'zuco2': set(range(100, 120))}
    return Layout(records, zone, set(), {}, block=15, buffer=5, minimum=10)


def build(seed=3, drift=0., length_effect=0., frequency_effect=0.):
    grid = layout()
    rng = np.random.default_rng(seed)
    popularity = 1 / np.arange(1, TYPES + 1)
    popularity /= popularity.sum()
    lengths = rng.integers(2, 12, TYPES)
    tokens = [rng.choice(TYPES, rng.integers(6, 14), p=popularity) for _ in grid.identities]
    vectors = rng.normal(size=(TYPES, DIMENSION))
    vectors[:, 0] += length_effect * (lengths - 6.5)
    vectors[:, 1] += frequency_effect * np.log(popularity / popularity.mean())
    reader, sentence, slot = [], [], []
    for r in range(len(grid.readers)):
        for s, row in enumerate(tokens):
            for p in range(len(row)):
                if rng.random() < .7:
                    reader.append(r)
                    sentence.append(s)
                    slot.append(p)
    reader, sentence, slot = np.array(reader), np.array(sentence), np.array(slot)
    row_type = np.array([tokens[s][p] for s, p in zip(sentence, slot)])
    durations = lengths[row_type] * 5 + rng.normal(scale=1, size=len(reader))
    data = WordData(grid, reader, sentence, slot, durations, rng.integers(1, 3, len(reader)).astype(float), tokens, vectors, lengths.astype(float))
    if drift:
        place = data.sentence_position
        table = data.type_vectors
        # Topic drifts with list position, through the words each sentence uses.
        data.candidates = data.candidates.copy()
        data.candidates[:, 0] += drift * np.sin(place / 8)
        data.candidates[:, 1] += drift * np.cos(place / 8)
        assert table is vectors
    return data, rng


def noisy(data, signal, rng, scale=3.):
    shift = rng.normal(size=(len(data.layout.readers), signal.shape[1]))[data.reader]
    return signal + scale * rng.normal(size=signal.shape) + shift


def label(data, real):
    waves = real.copy()
    return run_test(data, real, waves, None, [2, 5], PLAN, status=lambda m: None, iterations=500, reader_iterations=20)


def test_sentence_means_weigh_slots_equally_and_ignore_extra_readers():
    predicted = np.array([[1.], [3.], [10.], [0.], [5.]])
    sentence = np.array([7, 7, 7, 9, 9])
    slot = np.array([0, 0, 1, 0, 0])
    means, unique = sentence_means(predicted, sentence, slot)
    assert unique.tolist() == [7, 9]
    assert means[:, 0].tolist() == [6., 2.5]
    weighted, _ = sentence_means(predicted, sentence, slot, np.array([0., 1, 1, 1, 1]))
    assert weighted[0, 0] == 6.5


def test_gram_solver_matches_direct_ridge():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(200, 5)), rng.normal(size=(200, 2))
    solver = GramSolver(x)
    predicted = solver.predict(solver.project(x), solver.path(y), 10.)
    z = (x - x.mean(0)) / x.std(0)
    direct = y.mean(0) + z @ np.linalg.solve(z.T @ z + 10 * np.eye(5), z.T @ (y - y.mean(0)))
    assert np.allclose(predicted, direct)


def test_scaling_ignores_held_out_rows_and_zeroes_missing_values():
    x = np.random.default_rng(0).normal(size=(40, 3))
    readers = np.repeat([0, 1], 20)
    fit = np.ones(40, dtype=bool)
    fit[[0, 20]] = False
    changed = x.copy()
    changed[[0, 20]] += 1000
    assert np.allclose(scale_rows(x, readers, fit)[fit], scale_rows(changed, readers, fit)[fit])
    x[5, 1] = np.nan
    assert scale_rows(x, readers, fit)[5, 1] == 0
    with pytest.raises(ValueError):
        scale_rows(x, readers, readers == 0)


def test_shuffled_table_changes_training_words_and_keeps_held_out_candidates():
    data, _ = build()
    table = shuffled_table(data, 0, 42)
    fold = data.layout.folds[0]
    assert all(table.tokens[s] is data.tokens[s] for s in fold.sentences)
    train = np.flatnonzero(data.train_sentences(0))
    assert any(table.tokens[s] is not data.tokens[s] for s in train)
    assert np.array_equal(table.candidates[fold.sentences], data.candidates[fold.sentences])


def test_noise_keeps_each_observations_mean_and_removes_word_differences():
    data, rng = build()
    x = rng.normal(size=(len(data.reader), 4)).astype(np.float32)
    x[:, 0] += data.reader.astype(np.float32) * 3
    noise = noise_options(data, x, 1)(0)[0]
    rows = (data.reader == 2) & (data.sentence == 5)
    assert abs(noise[rows, 0].mean() - x[rows, 0].mean()) < 3 * x[rows, 0].std()
    target = data.type_vectors[data.row_type][:, 0]
    assert abs(np.corrcoef(noise[:, 1], target)[0, 1]) < .05
    assert abs(noise[:, 0].std() - x[:, 0].std()) < .2 * x[:, 0].std()


def test_new_reader_fit_never_uses_scored_readers(monkeypatch):
    data, rng = build()
    fit, score = half_readers(data, 42)
    assert not set(data.reader[fit]) & set(data.reader[score])
    x = rng.normal(size=(len(data.reader), 4))
    fitted = []
    original = GramSolver.__init__

    def spy(self, matrix):
        fitted.append(len(matrix))
        original(self, matrix)
    monkeypatch.setattr(GramSolver, '__init__', spy)
    rows, _, _ = run_arm(data, lambda i: [x], lambda i: [Table(data)], [1.], 42, fit_readers=fit, score_readers=score)
    assert set(data.reader[rows]) == set(data.reader[score])
    assert max(fitted) <= fit.sum()


def test_folds_only_score_their_own_sentences():
    data, rng = build()
    x = rng.normal(size=(len(data.reader), 4))
    rows, predicted, _ = run_arm(data, lambda i: [x], lambda i: [Table(data)], [1.], 42)
    assert len(rows) == len(np.unique(rows)) == len(predicted[0])
    assert set(data.sentence[rows]) == set(data.layout.scored.tolist())
    assert 0 < seen_type_share(data) <= 1
    assert WordScored(data, rows).average(predicted[0]).shape == (len(data.layout.scored), DIMENSION)


def test_planted_word_signal_is_recovered_and_improves_with_reader_count():
    data, rng = build()
    result = label(data, noisy(data, data.type_vectors[data.row_type], rng))
    curve = result['secondary']['reader_curve']['real']
    assert result['decision']['label'] == FIRST
    assert curve['all'] > curve['2'] > curve['1']
    assert result['secondary']['direct_word_test']['real'] > .2


def test_noise_stays_at_chance_at_every_reader_count():
    data, rng = build()
    result = label(data, noisy(data, np.zeros((len(data.reader), DIMENSION)), rng))
    assert result['decision']['label'] == THIRD
    assert all(abs(v - .1) < .05 for v in result['secondary']['reader_curve']['real'].values())


def test_sentence_level_signal_is_not_counted_as_word_locked():
    data, rng = build()
    result = label(data, noisy(data, data.candidates[data.sentence] * 3, rng, scale=1.))
    assert result['decision']['primary']['real'] > .3
    assert result['decision']['label'] != FIRST


@pytest.mark.parametrize('seed', SEEDS)
def test_position_signal_cannot_earn_the_first_reading(seed):
    data, rng = build(seed=seed, drift=3.)
    place = data.sentence_position[data.sentence]
    signal = np.c_[np.sin(place / 8) * 3, np.cos(place / 8) * 3, np.zeros((len(place), DIMENSION - 2))]
    assert label(data, noisy(data, signal, rng, scale=1.))['decision']['label'] != FIRST


@pytest.mark.parametrize('seed', SEEDS)
def test_length_signal_cannot_earn_the_first_reading(seed):
    data, rng = build(seed=seed, length_effect=1.)
    signal = np.c_[data.type_lengths[data.row_type], np.zeros((len(data.reader), DIMENSION - 1))]
    assert label(data, noisy(data, signal, rng, scale=1.))['decision']['label'] != FIRST


@pytest.mark.parametrize('seed', SEEDS)
def test_frequency_signal_cannot_earn_the_first_reading(seed):
    data, rng = build(seed=seed, frequency_effect=1.)
    counts = np.bincount(data.row_type, minlength=TYPES)
    signal = np.c_[np.log1p(counts[data.row_type]), np.zeros((len(data.reader), DIMENSION - 1))]
    assert label(data, noisy(data, signal, rng, scale=1.))['decision']['label'] != FIRST


@pytest.mark.parametrize('seed', SEEDS)
def test_duration_signal_cannot_earn_the_first_reading(seed):
    data, rng = build(seed=seed, length_effect=1.)
    signal = np.c_[data.durations, np.zeros((len(data.reader), DIMENSION - 1))]
    assert label(data, noisy(data, signal, rng, scale=1.))['decision']['label'] != FIRST


def test_reading_rules():
    good = {'sentence': .01, 'block': .01, 'reader': .01}
    lower = {name: good for name in ('gaussian', 'class_shuffled', 'gaze_word', 'position', 'partial')}
    gaps = {name: .03 for name in lower}
    assert read_outcome(gaps, lower)['label'] == FIRST
    assert read_outcome(dict(gaps, gaze_word=.01), lower)['label'] == SECOND
    assert read_outcome(dict(gaps, gaussian=.01), lower)['label'] == THIRD
    assert read_outcome(gaps, dict(lower, partial=dict(good, block=-.01)))['label'] == SECOND
    assert read_outcome(gaps, dict(lower, class_shuffled=dict(good, reader=-.01)))['label'] == THIRD


def test_band_vector_needs_every_band_and_drops_the_channel():
    keep = np.arange(104)
    values = {f'TRT_{b}': np.full((1, 105), 1.) for b in ('t1', 't2', 'a1', 'a2', 'b1', 'b2', 'g1', 'g2')}
    assert band_vector(values, keep).shape == (832,)
    assert np.allclose(band_vector(values, keep), np.log(2))
    missing = dict(values, TRT_g2=np.empty((0, 0)))
    assert band_vector(missing, keep) is None
    nan = dict(values, TRT_a1=np.full((1, 105), np.nan))
    assert band_vector(nan, keep) is None
    with pytest.raises(ValueError):
        band_vector(dict(values, TRT_a1=np.full((1, 105), -1.)), keep)


def test_waveform_bins_stop_when_the_fixation_ends():
    keep = np.arange(104)
    raw = np.ones((105, 60))
    bins = waveform_bins(raw, keep).reshape(6, 104)
    assert (bins[:2] == 1).all()
    assert np.isfinite(bins[2]).all() and np.isnan(bins[3:]).all()
    assert np.isnan(waveform_bins(np.array([[np.nan]]), keep)).all()
    assert np.isnan(waveform_bins(None, keep)).all()


def test_tokens_lose_edge_punctuation_and_case_only():
    assert runner.normal_token('"Henry,') == 'henry'
    assert runner.normal_token("Ford's") == "ford's"
    assert runner.normal_token('--') == '--'
