"""SYNTHETIC fixtures only. These test the confirmation v10 software, not EEG performance."""
from dataclasses import fields
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from eeg_decoding.data.dataset import Record
from eeg_decoding.evaluation import averaged_eeg, first_fixation, word_locked
from eeg_decoding.evaluation.confirmation import (CONFIRMED, CONTRADICTED, INCONCLUSIVE, ConfirmLayout, ConfirmWordData, confirm_audit, read_claims,
                                                  reserved_blocks)
from eeg_decoding.evaluation.word_locked import WordData

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v5_tests = load('averaged_fixtures', 'tests/test_averaged_eeg.py')
v6_tests = load('word_locked_fixtures', 'tests/test_word_locked.py')
v9_tests = load('first_fixation_fixtures', 'tests/test_first_fixation.py')
# The frozen plan's check used 20 seeds; the everyday suite keeps 5 for speed.
SEEDS = range(int(os.environ.get('CONFIRMATION_SEEDS', 5)))
BLOCK, BUFFER, SENTENCES = 15, 5, 120
BLOCKS = {'zuco1': [list(range(30, 45)), list(range(80, 95))], 'zuco2': [list(range(35, 50)), list(range(85, 100))]}


def zone_of(blocks):
    return {release: {i + d for run in runs for i in run for d in range(-BUFFER, BUFFER + 1) if 0 <= i + d < SENTENCES} for release, runs in blocks.items()}


def records_for(text, readers):
    records = []
    for release, offset in (('zuco1', 0), ('zuco2', 1000)):
        for reader in range(readers):
            for i in range(SENTENCES):
                records.append(Record(release, f'{release[-1]}R{reader}', '1', 'nr', str(i), text(offset + i), f'{release}_{reader}_{i}.npz', ['C3'], 200.,
                                      provenance={'synthetic': True}))
    return records


def confirm_layout(readers=8, v1_test=(), text=v5_tests.sentence):
    records = records_for(text, readers)
    return records, ConfirmLayout(records, BLOCKS, zone_of(BLOCKS), set(v1_test), {}, block=BLOCK, buffer=BUFFER, minimum=10)


def test_blocks_come_from_the_v2_split_with_their_buffers():
    records = records_for(v5_tests.sentence, 2)
    blocks, zone = reserved_blocks(records, seed=42, block=BLOCK, buffer=BUFFER, val_blocks=1, test_blocks=2)
    for release, runs in blocks.items():
        assert len(runs) == 2 and all(len(run) == BLOCK and run == list(range(run[0], run[0] + BLOCK)) for run in runs)
        for run in runs:
            assert set(range(max(run[0] - BUFFER, 0), min(run[-1] + BUFFER, SENTENCES - 1) + 1)) <= zone[release]


def test_every_fold_trains_outside_all_test_blocks_and_buffers():
    _, layout = confirm_layout()
    assert len(layout.folds) == 4 and confirm_audit(layout, 42) == []
    tested = np.isin(layout.sentence, layout.test_sentences)
    for index in range(4):
        assert not (layout.train_mask(index) & tested).any()
    present = {(r.dataset, int(r.sentence_id)) for r in layout.rows}
    for release, ids in zone_of(BLOCKS).items():
        buffers = ids - {i for run in BLOCKS[release] for i in run}
        assert not {(release, i) for i in buffers} & present
    assert {layout.identities[s] for s in layout.scored} == {(rel, i) for rel, runs in BLOCKS.items() for run in runs for i in run}


def test_one_fit_serves_every_block():
    _, layout = confirm_layout()
    first = [(b.release, b.low, b.high) for b in layout.inner_blocks(0, 42)]
    for index in range(1, 4):
        assert [(b.release, b.low, b.high) for b in layout.inner_blocks(index, 42)] == first
        assert (layout.train_mask(index) == layout.train_mask(0)).all()


def test_v1_test_sentences_in_a_test_block_are_neither_fitted_nor_scored():
    closed = {('zuco1', 31), ('zuco2', 90)}
    _, layout = confirm_layout(v1_test=closed)
    present = {(r.dataset, int(r.sentence_id)) for r in layout.rows}
    assert not closed & present
    assert not closed & {layout.identities[s] for s in layout.scored}


def test_the_audit_catches_a_leaking_fold():
    _, layout = confirm_layout()
    layout.train_mask = lambda index: np.ones(len(layout.rows), dtype=bool)
    assert any('test-block row is in training' in f for f in confirm_audit(layout, 42))


def test_shuffled_tables_keep_test_targets_out_of_training():
    _, layout = confirm_layout()
    text = v5_tests.smooth_text(layout)
    for table in averaged_eeg.shuffled_tables(layout, 0, text, 3, 42):
        assert np.array_equal(table[layout.test_sentences], text[layout.test_sentences])


def sentence_label(layout, real, seed=9):
    text = v5_tests.smooth_text(layout)
    durations = np.array([len(r.text.split()) for r in layout.rows], dtype=float)
    return v5_tests.label_for(layout, text, real, durations, seed)


def sentence_gap_clear(decision):
    gap = decision['gaps']['gaussian']
    bounds = [decision['intervals']['gaussian']['sentence_interval'], decision['reader_intervals']['gaussian']]
    return read_claims({'v5': {'gap': gap, 'bounds': bounds}}, {'gap': 0., 'bounds': [(-1, 1)]}, {'gap': 0., 'bounds': [(-1, 1)]})['C1'] == CONTRADICTED


def test_planted_sentence_signal_contradicts_c1():
    _, layout = confirm_layout()
    text = v5_tests.smooth_text(layout)
    real = v5_tests.noisy(layout, text[layout.sentence] @ np.random.default_rng(4).normal(size=(text.shape[1], 24)), scale=2.)
    assert sentence_gap_clear(sentence_label(layout, real)['decision'])


@pytest.mark.parametrize('seed', SEEDS)
def test_position_or_duration_alone_never_contradicts_c1(seed):
    _, layout = confirm_layout()
    rng = np.random.default_rng(seed)
    durations = np.array([len(r.text.split()) for r in layout.rows], dtype=float)
    position = np.stack([np.sin(layout.position / 9), np.cos(layout.position / 9), layout.position / 100.], 1)
    for signal in (position, durations[:, None]):
        real = v5_tests.noisy(layout, signal @ rng.normal(size=(signal.shape[1], 24)) * 3, scale=2., seed=seed)
        assert not sentence_gap_clear(sentence_label(layout, real, seed=seed + 50)['decision'])


def word_data(seed=3):
    """The v6 synthetic word rows over a confirmation layout."""
    original = v6_tests.layout
    v6_tests.layout = lambda: confirm_layout(readers=6, text=lambda i: f's{i} x')[1]
    try:
        data, rng = v6_tests.build(seed)
    finally:
        v6_tests.layout = original
    values = {f.name: getattr(data, f.name) for f in fields(WordData)}
    return ConfirmWordData(**values), rng


def test_word_rows_train_outside_every_test_block():
    data, _ = word_data()
    for index in range(4):
        train = data.train_sentences(index)
        assert not train[data.layout.test_sentences].any() and train.sum() == len(data.layout.identities) - len(data.layout.test_sentences)
    table = word_locked.shuffled_table(data, 0, 7)
    assert np.allclose(table.candidates[data.layout.test_sentences], data.candidates[data.layout.test_sentences])


def test_planted_word_signal_is_found_and_noise_is_not():
    data, rng = word_data()
    signal = data.type_vectors[data.row_type]
    planted = v9_tests.label(data, v9_tests.waves_from(data, signal, rng))['decision']
    assert planted['gaps']['gaussian'] >= .02 and planted['lower_bounds']['gaussian']['sentence'] > 0
    nothing = v9_tests.label(data, v9_tests.waves_from(data, np.zeros_like(signal), rng))['decision']
    assert not (nothing['gaps']['gaussian'] >= .02 and nothing['lower_bounds']['gaussian']['sentence'] > 0)


@pytest.mark.parametrize('seed', SEEDS)
def test_duration_or_length_alone_never_contradicts_c1_word_level(seed):
    data, rng = word_data(seed)
    for signal in (data.durations[:, None], data.type_lengths[data.row_type][:, None]):
        decision = v9_tests.label(data, v9_tests.waves_from(data, signal, rng))['decision']
        bounds = [decision['intervals']['gaussian']['sentence_interval'], decision['reader_intervals']['gaussian']]
        assert read_claims({'v9': {'gap': decision['gaps']['gaussian'], 'bounds': bounds}}, {'gap': 0., 'bounds': [(-1, 1)]},
                           {'gap': 0., 'bounds': [(-1, 1)]})['C1'] == CONFIRMED


def test_reading_rules():
    quiet = {'gap': .005, 'bounds': [(-.02, .03), (-.01, .02)]}
    loud = {'gap': .05, 'bounds': [(.01, .09), (.02, .08)]}
    small_but_sure = {'gap': .03, 'bounds': [(.01, .05)]}
    assert read_claims({'a': quiet, 'b': small_but_sure}, {'gap': -.10, 'bounds': [(-.15, -.05)]}, {'gap': .001, 'bounds': [(-.02, .025), (-.02, .02)]}) == \
        {'C1': CONTRADICTED, 'C2': CONFIRMED, 'C3': CONFIRMED}
    assert read_claims({'a': quiet}, {'gap': .05, 'bounds': [(.01, .09)]}, loud) == {'C1': CONFIRMED, 'C2': CONTRADICTED, 'C3': CONTRADICTED}
    assert read_claims({'a': quiet}, {'gap': -.05, 'bounds': [(-.12, .01)]}, {'gap': .01, 'bounds': [(-.02, .035), (-.01, .02)]}) == \
        {'C1': CONFIRMED, 'C2': INCONCLUSIVE, 'C3': INCONCLUSIVE}
    # A gap under 2 points never counts as clear, even with every bound above 0.
    assert read_claims({'a': {'gap': .015, 'bounds': [(.001, .03)]}}, {'gap': 0., 'bounds': [(-1, 1)]}, {'gap': 0., 'bounds': [(-1, 1)]})['C1'] == CONFIRMED
