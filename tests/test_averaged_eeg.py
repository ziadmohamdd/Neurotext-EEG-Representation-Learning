"""SYNTHETIC fixtures only. These test the averaged-EEG software, not EEG performance."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from eeg_decoding.data.dataset import Record
from eeg_decoding.evaluation import averaged_eeg as averaged
from eeg_decoding.evaluation.averaged_eeg import (FIRST, Layout, Scored, THIRD, fold_audit, half_readers, read_outcome, resample_gap, run_arm,
                                                  run_test, scale_by_reader, shuffled_tables)
from eeg_decoding.training.alignment_train import TextCache

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('averaged_v5', ROOT/'scripts/run_averaged_eeg.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
PLAN = {'seed': 42, 'model': {'alphas': [1, 100, 10000]}, 'arms_shuffles': 2, 'decision_rule': {'minimum_gap_points': 2.},
        'averaging': {'reader_counts_for_curve': [1, 2, 'all'], 'subset_draws_count': 2}}
SENTENCES, READERS, DIMENSION = 120, 8, 16


def sentence(index):
    return " ".join(f"w{index}x{p}" for p in range(5 + index % 9))


def build(zone=None, v1_test=(), forbidden=None):
    records = []
    for release, offset in (('zuco1', 0), ('zuco2', 1000)):
        for reader in range(READERS):
            for i in range(SENTENCES):
                records.append(Record(release, f'{release[-1]}R{reader}', '1', 'nr', str(i), sentence(offset + i), f'{release}_{reader}_{i}.npz',
                                      ['C3'], 200., provenance={'synthetic': True}))
    zone = zone or {'zuco1': set(range(100, 120)), 'zuco2': set(range(100, 120))}
    return records, Layout(records, zone, set(v1_test), forbidden or {}, block=15, buffer=5, minimum=10)


def smooth_text(layout, drift=0., seed=3):
    rng = np.random.default_rng(seed)
    text = rng.normal(size=(len(layout.identities), DIMENSION))
    if drift:
        for number, (release, i) in enumerate(layout.identities):
            text[number, 0] += drift * np.sin(i / 8)
            text[number, 1] += drift * np.cos(i / 8)
    return text


def noisy(layout, signal, scale=4., seed=1):
    rng = np.random.default_rng(seed)
    reader_shift = rng.normal(size=(len(layout.readers), signal.shape[1]))[layout.reader]
    return signal + scale * rng.normal(size=signal.shape) + reader_shift


def label_for(layout, text, real, durations, seed=9):
    pure = noisy(layout, np.zeros_like(real), seed=seed)
    features = {'real': real, 'gaussian': pure, 'binned_real': real, 'binned_gaussian': pure}
    return run_test(layout, text, features, durations, [2, 5], PLAN, status=lambda m: None, iterations=500, reader_iterations=30)


def test_folds_stay_outside_protected_and_keep_the_buffer():
    records, layout = build()
    assert not {i for release, i in layout.identities if i >= 100}
    for index, fold in enumerate(layout.folds):
        train = layout.train_mask(index)
        same = train & (layout.release == fold.release)
        assert np.abs(layout.position[same] - fold.low).min() > layout.buffer or layout.position[same].min() > fold.high + layout.buffer
        assert not np.isin(layout.sentence[train], fold.sentences).any()


def test_a_block_never_crosses_the_zone_and_small_blocks_are_not_held_out():
    records, layout = build(zone={'zuco1': set(range(40, 60)), 'zuco2': set()}, forbidden={'zuco2': set(range(0, 10))})
    for fold in layout.folds:
        if fold.release == 'zuco1':
            assert fold.high < 40 or fold.low >= 60
        assert len(fold.sentences) >= 10
    assert all(not (f.release == 'zuco2' and f.low == 0) for f in layout.folds)


def test_inner_blocks_avoid_the_fold_and_its_buffer():
    records, layout = build()
    for index, fold in enumerate(layout.folds):
        for inner in layout.inner_blocks(index, 42):
            if inner.release == fold.release:
                assert inner.high < fold.low - layout.buffer or inner.low > fold.high + layout.buffer
            assert inner.high < 100


def test_scaling_ignores_held_out_rows():
    x = np.random.default_rng(0).normal(size=(40, 3))
    readers = np.repeat([0, 1], 20)
    fit = np.ones(40, dtype=bool)
    fit[[0, 20]] = False
    changed = x.copy()
    changed[[0, 20]] += 1000
    assert np.allclose(scale_by_reader(x, readers, fit)[fit], scale_by_reader(changed, readers, fit)[fit])
    with pytest.raises(ValueError):
        scale_by_reader(x, readers, readers == 0)


def test_shuffled_tables_keep_held_out_targets_out_of_training():
    records, layout = build()
    text = smooth_text(layout)
    table = shuffled_tables(layout, 0, text, 1, 42)[0]
    train = layout.train_mask(0)
    trained = np.unique(layout.sentence[train])
    held = layout.folds[0].sentences
    assert not (table[trained] == text[held][:, None]).all(-1).any()
    assert (table[held] == text[held]).all()
    assert (table[trained] != text[trained]).any()


def test_planted_signal_is_recovered_and_improves_with_reader_count():
    records, layout = build()
    text = smooth_text(layout)
    real = noisy(layout, text[layout.sentence])
    result = label_for(layout, text, real, np.random.default_rng(2).normal(size=len(layout.rows)))
    curve = result['secondary']['reader_curve']['real']
    assert result['decision']['label'] == FIRST
    assert curve['all'] > curve['2'] > curve['1']
    assert result['decision']['primary']['real'] > .5


def test_noise_stays_at_chance_at_every_reader_count():
    records, layout = build()
    text = smooth_text(layout)
    real = noisy(layout, np.zeros((len(layout.rows), DIMENSION)))
    result = label_for(layout, text, real, np.random.default_rng(2).normal(size=len(layout.rows)))
    assert result['decision']['label'] == THIRD
    assert all(abs(v - .1) < .05 for v in result['secondary']['reader_curve']['real'].values())


@pytest.mark.parametrize('seed', [1, 2, 3, 4, 5])
def test_position_only_signal_cannot_earn_the_first_reading(seed):
    records, layout = build()
    text = smooth_text(layout, drift=3., seed=seed)
    position = np.array([[np.sin(i / 8), np.cos(i / 8)] for _, i in (layout.identities[s] for s in layout.sentence)])
    real = noisy(layout, np.c_[position * 3, np.zeros((len(layout.rows), DIMENSION - 2))], scale=1., seed=seed)
    result = label_for(layout, text, real, np.random.default_rng(seed).normal(size=len(layout.rows)), seed=seed + 50)
    assert result['decision']['label'] != FIRST


@pytest.mark.parametrize('seed', [1, 2, 3, 4, 5])
def test_duration_only_signal_cannot_earn_the_first_reading(seed):
    records, layout = build()
    text = smooth_text(layout, seed=seed)
    words = np.array([len(r.text.split()) for r in layout.rows], dtype=float)
    for number, (release, i) in enumerate(layout.identities):
        text[number, 0] += 2 * (5 + (i + (1000 if release == 'zuco2' else 0)) % 9 - 9)
    real = noisy(layout, np.c_[words, np.zeros((len(layout.rows), DIMENSION - 1))], scale=1., seed=seed)
    durations = words + np.random.default_rng(seed).normal(scale=.1, size=len(words))
    result = label_for(layout, text, real, durations, seed=seed + 50)
    assert result['decision']['label'] != FIRST


def test_new_reader_fit_never_uses_scored_readers(monkeypatch):
    records, layout = build()
    fit, score = half_readers(layout, 42)
    assert not set(layout.reader[fit]) & set(layout.reader[score])
    text = smooth_text(layout)
    x = np.random.default_rng(0).normal(size=(len(layout.rows), 4))
    masks = []
    original = averaged._fit_rows

    def spy(features, sentence, chosen, tables, fit_means):
        masks.append(chosen.copy())
        return original(features, sentence, chosen, tables, fit_means)
    monkeypatch.setattr(averaged, '_fit_rows', spy)
    rows, _, _ = run_arm(layout, lambda train: [x], lambda i: [text], [1.], 42, fit_readers=fit, score_readers=score)
    assert set(layout.reader[rows]) == set(layout.reader[score])
    assert masks and all(not set(layout.reader[m]) & set(layout.reader[score]) for m in masks)
    assert Scored(layout, rows, text).count == len(layout.scored)


def test_reading_rules():
    good = {'sentence': .01, 'block': .01, 'reader': .01}
    lower = {name: good for name in ('gaussian', 'class_shuffled', 'timing', 'position', 'partial')}
    gaps = {'gaussian': .03, 'class_shuffled': .03, 'timing': .03, 'position': .03, 'partial': .03}
    assert read_outcome(gaps, lower)['label'] == FIRST
    assert read_outcome(dict(gaps, position=.01), lower)['label'] == 'The gain does not exceed timing or position'
    assert read_outcome(dict(gaps, gaussian=.01), lower)['label'] == THIRD
    assert read_outcome(gaps, dict(lower, gaussian=dict(good, block=-.01)))['label'] == THIRD
    assert read_outcome(gaps, dict(lower, partial=dict(good, sentence=-.01)))['label'] == 'The gain does not exceed timing or position'


def test_block_interval_is_wider_when_blocks_differ():
    gap = np.r_[np.full(50, .2), np.full(50, -.1)]
    report = resample_gap(gap, np.repeat([0, 1], 50), 2000, 1)
    assert report['block_interval'][0] < report['sentence_interval'][0]


def test_audit_catches_a_protected_row_and_a_near_copy(tmp_path):
    records, layout = build()
    keys = sorted({r.key for r in records})
    vectors = np.random.default_rng(3).normal(size=(len(keys), 64)).astype(np.float32)
    path = tmp_path/'SYNTHETIC_text.npz'
    np.savez_compressed(path, keys=np.array(keys), embeddings=vectors, metadata=json.dumps({'synthetic': True}))
    text = TextCache(path)
    closed = {('zuco1', i) for i in range(100, 120)} | {('zuco2', i) for i in range(100, 120)}
    assert fold_audit(records, layout, text, closed, set())['failures'] == []
    assert fold_audit(records, layout, text, closed | {('zuco1', 0)}, set())['failures']
    held = layout.key_of[layout.folds[0].sentences[0]]
    far = layout.rows[int(np.flatnonzero(layout.train_mask(0))[0])].key
    data = dict(np.load(path, allow_pickle=False))
    data['embeddings'][keys.index(far)] = data['embeddings'][keys.index(held)]
    copy = tmp_path/'SYNTHETIC_copy.npz'
    np.savez_compressed(copy, **data)
    assert any('nearly matches' in f for f in fold_audit(records, layout, TextCache(copy), closed, set())['failures'])


def test_features_keep_one_duration_and_never_normalize_by_reader(tmp_path):
    rows = []
    rng = np.random.default_rng(1)
    for i, samples in enumerate((400, 900)):
        path = tmp_path/f'SYNTHETIC_{i}.npz'
        np.savez_compressed(path, eeg=rng.normal(size=(3, samples)).astype(np.float32))
        rows.append(Record('zuco1', 'R0', '1', 'nr', str(i), sentence(i), str(path), ['C3', 'Cz', 'C4'], 200., provenance={'units': 'uV', 'synthetic': True}))
    pipeline = {'notch_hz': 50, 'drop_channels': ['Cz'], 'window_samples': 600}
    features, durations = runner.build_features(rows, pipeline, 42, status=lambda m: None)
    assert durations.tolist() == [2., 4.5]
    assert features['binned_real'].shape == (2, 2 * runner.BINS)
    assert features['real'].shape == features['gaussian'].shape
    assert not np.allclose(features['real'], features['gaussian'])
    again, _ = runner.build_features(rows, pipeline, 42, status=lambda m: None)
    assert np.array_equal(features['gaussian'], again['gaussian'])
