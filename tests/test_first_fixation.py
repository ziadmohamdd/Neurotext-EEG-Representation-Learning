"""SYNTHETIC fixtures only. These test the first-fixation software, not EEG performance."""
import importlib.util
import os
from pathlib import Path
import numpy as np
import pytest
from eeg_decoding.evaluation.first_fixation import FIRST, SECOND, THIRD, read_outcome, run_test, without_channels

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('word_locked_fixtures', ROOT/'tests/test_word_locked.py')
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
# The frozen plan's check used 20 seeds; the everyday suite keeps 5 for speed.
SEEDS = range(int(os.environ.get('FIRST_FIXATION_SEEDS', 5)))
PLAN = dict(fixtures.PLAN, sentence_prediction={'reader_counts_for_curve': [1, 'all'], 'subset_draws_count': 2})
CHANNELS, BINS, ANTERIOR = 8, 6, np.array([0, 1])
WIDTH = CHANNELS * BINS


def waves_from(data, signal, rng, scale=3.):
    """Spread a signal over every channel and bin, then add reader offsets and noise."""
    return fixtures.noisy(data, signal @ rng.normal(size=(signal.shape[1], WIDTH)), rng, scale)


build = fixtures.build


def label(data, waves):
    return run_test(data, waves, CHANNELS, ANTERIOR, PLAN, status=lambda m: None, iterations=500, reader_iterations=20)


def only_anterior(data, signal, rng, scale=3.):
    """A signal that appears only in the anterior channels' columns."""
    waves = waves_from(data, np.zeros_like(signal), rng, scale)
    columns = np.flatnonzero(np.isin(np.arange(WIDTH) % CHANNELS, ANTERIOR))
    waves[:, columns] += signal @ rng.normal(size=(signal.shape[1], len(columns))) * 3
    return waves


def test_removing_channels_drops_every_bin_of_those_channels():
    waves = np.arange(2 * WIDTH, dtype=float).reshape(2, WIDTH)
    kept = without_channels(waves, CHANNELS, ANTERIOR)
    assert kept.shape == (2, WIDTH - BINS * len(ANTERIOR))
    assert not np.isin(np.arange(WIDTH)[np.isin(np.arange(WIDTH) % CHANNELS, ANTERIOR)], kept[0]).any()


def test_planted_waveform_signal_earns_the_first_label():
    data, rng = build()
    result = label(data, waves_from(data, data.type_vectors[data.row_type], rng))
    assert result['decision']['label'] == FIRST


def test_no_signal_does_not_hold():
    data, rng = build()
    result = label(data, waves_from(data, np.zeros((len(data.reader), fixtures.DIMENSION)), rng))
    assert result['decision']['label'] == THIRD


def test_sentence_level_signal_is_not_counted_as_waveform():
    data, rng = build()
    result = label(data, waves_from(data, data.candidates[data.sentence] * 3, rng, scale=1.))
    assert result['decision']['label'] != FIRST


@pytest.mark.parametrize('seed', SEEDS)
def test_duration_signal_cannot_earn_the_first_label(seed):
    data, rng = build(seed=seed, length_effect=1.)
    signal = np.c_[data.durations, np.zeros((len(data.reader), fixtures.DIMENSION - 1))]
    assert label(data, waves_from(data, signal, rng, scale=1.))['decision']['label'] != FIRST


@pytest.mark.parametrize('seed', SEEDS)
def test_length_signal_cannot_earn_the_first_label(seed):
    data, rng = build(seed=seed, length_effect=1.)
    signal = np.c_[data.type_lengths[data.row_type], np.zeros((len(data.reader), fixtures.DIMENSION - 1))]
    assert label(data, waves_from(data, signal, rng, scale=1.))['decision']['label'] != FIRST


@pytest.mark.parametrize('seed', SEEDS)
def test_anterior_only_signal_cannot_earn_the_first_label(seed):
    data, rng = build(seed=seed)
    assert label(data, only_anterior(data, data.type_vectors[data.row_type], rng))['decision']['label'] != FIRST


def test_reading_rules():
    good = {'sentence': .01, 'block': .01, 'reader': .01}
    lower = {name: good for name in ('gaussian', 'class_shuffled', 'adds_to_gaze', 'not_eye_channels', 'partial')}
    gaps = {name: .03 for name in lower}
    assert read_outcome(gaps, lower)['label'] == FIRST
    assert read_outcome(dict(gaps, adds_to_gaze=.01), lower)['label'] == SECOND
    assert read_outcome(gaps, dict(lower, not_eye_channels=dict(good, reader=-.01)))['label'] == SECOND
    assert read_outcome(dict(gaps, gaussian=.01), lower)['label'] == THIRD
    assert read_outcome(gaps, dict(lower, class_shuffled=dict(good, block=-.01)))['label'] == THIRD
