"""SYNTHETIC fixtures only. These test the same-sentence software, not EEG performance."""
import os
import numpy as np
import pytest
from eeg_decoding.data.dataset import Record
from eeg_decoding.evaluation.same_sentence import (FIRST, Pairing, THIRD, block_effect, match, normal_text, read_outcome, reader_weights, remove, run_test,
                                                   shared_pairs, shuffled, standardize)

PLAN = {'seed': 42, 'evaluation': {'shuffles': 200, 'block_near_positions': 25, 'block_shuffles': 100}, 'decision_rule': {'minimum_gap_points': 2.}}
SHARED, ONLY, READERS, DIMENSION = 48, 40, 12, 40
SEEDS = int(os.environ.get('SAME_SENTENCE_SEEDS', 5))


def build(seed):
    """Two releases share SHARED texts at unrelated list positions; the rest appear in one release only."""
    rng = np.random.default_rng(seed)
    records, place = [], {}
    for release in ('zuco1', 'zuco2'):
        own = [f'only {release} text number {i}' for i in range(ONLY)]
        shared = [f'shared text number {i} ' + 'word ' * (i % 13) for i in range(SHARED)]
        listing = [own[i] if i < ONLY else shared[i - ONLY] for i in rng.permutation(ONLY + SHARED)]
        place[release] = {i: i for i in range(len(listing))}
        for reader in range(READERS):
            for i, text in enumerate(listing):
                records.append(Record(release, f'{release[-1]}R{reader}', '1', 'nr', str(i), text, f'{release}_{reader}_{i}.npz', ['C3'], 200., provenance={'synthetic': True}))
    releases, pairs = shared_pairs(records, set())
    return Pairing(records, pairs, releases, place)


def fixture(pairing, kind, seed, strength=.5):
    rng = np.random.default_rng(seed + 1000)
    rows = len(pairing.rows)
    real = rng.normal(size=(rows, DIMENSION))
    noise = rng.normal(size=(rows, DIMENSION))
    durations = pairing.words * .3 + rng.normal(scale=.2, size=rows)
    place = np.array([pairing.place[r.dataset][int(r.sentence_id)] for r in pairing.rows], dtype=float)
    weights = rng.normal(size=DIMENSION)
    if kind == 'planted':
        texts = {normal_text(r.text) for r in pairing.rows}
        signal = {t: rng.normal(size=DIMENSION) for t in sorted(texts)}
        real += strength * np.array([signal[normal_text(r.text)] for r in pairing.rows])
    elif kind == 'length':
        real += strength * np.outer((pairing.words - pairing.words.mean()) / pairing.words.std(), weights)
    elif kind == 'duration':
        real += strength * np.outer((durations - durations.mean()) / durations.std(), weights)
    elif kind == 'position':
        real += strength * np.outer((place - place.mean()) / place.std(), weights)
    return {'real': real, 'gaussian': noise}, durations


def reading(pairing, kind, seed, iterations=300, reader_iterations=40, strength=.5):
    features, durations = fixture(pairing, kind, seed, strength)
    return run_test(pairing, features, durations, PLAN, status=lambda m: None, iterations=iterations, reader_iterations=reader_iterations)


def test_shared_pairs_leave_out_protected_copies():
    make = lambda release, i, text: Record(release, 'R', '1', 'nr', str(i), text, 'x.npz', ['C3'], 200.)
    records = [make('zuco1', 0, 'Same, text.'), make('zuco2', 3, 'same text'), make('zuco1', 1, 'Closed text'), make('zuco2', 4, 'closed text'),
               make('zuco1', 2, 'Twice here'), make('zuco1', 7, 'twice here'), make('zuco2', 9, 'twice here'), make('zuco2', 5, 'Only one')]
    releases, pairs = shared_pairs(records, {('zuco2', 4)})
    assert releases == ['zuco1', 'zuco2']
    assert pairs == [(0, 3), (2, 9)]


def test_match_is_perfect_when_sides_agree_and_chance_when_shuffled():
    a = standardize(np.random.default_rng(1).normal(size=(48, 30)))
    assert match(a, a).mean() == pytest.approx(1.)
    assert shuffled(a, a, 2000, 1).mean() == pytest.approx(.10, abs=.01)


def test_order_scores_each_sentence_against_its_new_partner():
    a = standardize(np.random.default_rng(1).normal(size=(48, 30)))
    order = np.random.default_rng(2).permutation(48)
    assert match(a, a[np.argsort(order)], order=order).mean() == pytest.approx(1.)


def test_zero_vector_is_a_miss_not_a_crash():
    a = standardize(np.random.default_rng(1).normal(size=(12, 6)))
    a[3] = 0
    assert match(a, a)[3] < .6


def test_remove_takes_out_covariates():
    rng = np.random.default_rng(1)
    c = rng.normal(size=(48, 3))
    x = np.c_[c[:, 1] * 3, c[:, 2], c[:, 0] ** 2]
    assert np.abs(remove(x, c)).max() < 1e-8


def test_scaling_ignores_shared_rows():
    pairing = build(0)
    x = np.random.default_rng(1).normal(size=(len(pairing.rows), 4))
    changed = x.copy()
    changed[pairing.pair >= 0] += 100
    assert np.allclose(pairing.scaled(x)[pairing.outside], pairing.scaled(changed)[pairing.outside])


def test_reader_weights_change_the_average():
    pairing = build(0)
    x = np.random.default_rng(1).normal(size=(len(pairing.rows), 4))
    one = (pairing.reader == 0) * 1.
    a, b = pairing.means(x, one)
    assert not np.allclose(a, pairing.means(x)[0])
    members = (pairing.reader == 0) & (pairing.side == 0) & (pairing.pair == 0)
    assert a[0] == pytest.approx(x[members].mean(0))
    assert np.allclose(b, pairing.means(x)[1])


def test_reader_resampling_keeps_each_release_size():
    side = np.array([0] * 12 + [1] * 18)
    seen = np.zeros(30)
    for seed in range(200):
        weight = reader_weights(side, np.random.default_rng(seed))
        assert weight[:12].sum() == 12 and weight[12:].sum() == 18
        seen += weight
    assert (seen > 0).all()


def test_block_effect_finds_drift_and_not_noise():
    rng = np.random.default_rng(1)
    place = np.arange(150.)
    drift = np.c_[np.sin(place / 12), np.cos(place / 12)]
    x = rng.normal(size=(150, 30)) + 2 * drift @ rng.normal(size=(2, 30))
    assert block_effect(x, place, 25, 200, 1)['share_of_shuffles_at_least_this_large'] < .05
    assert block_effect(rng.normal(size=(150, 30)), place, 25, 200, 1)['near_minus_far'] == pytest.approx(0, abs=.01)


def test_outcome_needs_every_lower_bound():
    gaps = dict.fromkeys(('gaussian', 'class_shuffled', 'timing', 'position', 'partial'), .05)
    lower = {name: {'sentence': .01} for name in gaps}
    assert read_outcome(gaps, lower)['label'] == FIRST
    lower['gaussian']['reader'] = -.01
    assert read_outcome(gaps, lower)['label'] == THIRD


def test_planted_signal_earns_first_reading():
    passed = sum(reading(build(seed), 'planted', seed)['decision']['label'] == FIRST for seed in range(SEEDS))
    assert passed == SEEDS


@pytest.mark.parametrize('kind', ['none', 'length', 'duration', 'position'])
def test_structure_never_earns_first_reading(kind):
    for seed in range(SEEDS):
        assert reading(build(seed), kind, seed)['decision']['label'] != FIRST


def test_no_signal_stays_near_chance():
    assert reading(build(1), 'none', 1)['decision']['primary']['real'] == pytest.approx(.10, abs=.05)
