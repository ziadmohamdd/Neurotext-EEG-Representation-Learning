"""SYNTHETIC fixtures only. These test the public-outputs software, not any model's performance."""
import os
import numpy as np
import pytest
from eeg_decoding.evaluation.public_outputs import (FIRST, SECOND, THIRD, expected_scores, label_corpus, read_outcome, read_outputs, run_check)
from eeg_decoding.evaluation.retrieval import normalize

PLAN = {'seed': 42, 'decision_rule': {'minimum_gap_points': 2., 'resamples': 1000}}
SENTENCES, READERS, DIMENSION = 100, 12, 48
SEEDS = int(os.environ.get('PUBLIC_OUTPUTS_SEEDS', 5))


def fixture(kind, seed, strength=.6):
    """Two corpora with distinct mean text. Outputs carry nothing, the corpus only, or the sentence itself."""
    rng = np.random.default_rng(seed)
    corpus = np.array(['movie_review'] * 40 + ['biography'] * 60)
    centres = {name: rng.normal(size=DIMENSION) for name in ('movie_review', 'biography')}
    bank = normalize(np.stack([centres[c] for c in corpus]) + rng.normal(size=(SENTENCES, DIMENSION)))
    own = np.repeat(np.arange(SENTENCES), READERS)
    real = rng.normal(size=(len(own), DIMENSION))
    if kind == 'corpus':
        real += 2 * np.stack([centres[c] for c in corpus[own]])
    elif kind == 'sentence':
        real += strength * np.sqrt(DIMENSION) * bank[own]
    return {'real': normalize(real), 'noise': normalize(rng.normal(size=real.shape))}, own, bank, corpus


@pytest.mark.parametrize('kind,label', [('none', FIRST), ('corpus', SECOND), ('sentence', THIRD)])
def test_each_kind_of_output_earns_its_label(kind, label):
    for seed in range(SEEDS):
        vectors, own, bank, corpus = fixture(kind, seed)
        assert run_check(vectors, own, bank, corpus, PLAN)['decision']['label'] == label, seed


def test_corpus_only_output_scores_chance_within_a_corpus():
    vectors, own, bank, corpus = fixture('corpus', 0)
    scores = run_check(vectors, own, bank, corpus, PLAN)['decision']['scores']
    assert scores['real']['mixed'] > 15 and abs(scores['real']['within_corpus'] - 10) < 3


def test_label_only_predictor_cannot_pick_a_sentence_within_its_corpus():
    vectors, own, bank, corpus = fixture('none', 0)
    scores = run_check(vectors, own, bank, corpus, PLAN)['decision']['scores']['corpus_label_only']
    assert scores['mixed'] > 15 and scores['within_corpus'] < 10


def test_expected_score_is_exact_and_ties_are_misses():
    bank = np.eye(4)
    pool = np.ones((4, 4), dtype=bool)
    perfect = expected_scores(bank[[0]], np.array([0]), bank, pool, 2)
    tied = expected_scores(np.full((1, 4), .5), np.array([0]), bank, pool, 2)
    half = expected_scores(normalize(np.array([[1., 2., 0., 0.]])), np.array([0]), bank, pool, 2)
    assert perfect[0] == 1 and tied[0] == 0 and half[0] == pytest.approx(2 / 3)


def test_identical_outputs_for_every_row_score_at_or_below_chance():
    vectors, own, bank, corpus = fixture('none', 1)
    vectors['real'] = np.tile(vectors['real'][:1], (len(own), 1))
    scores = run_check(vectors, own, bank, corpus, PLAN)['decision']
    assert scores['label'] == FIRST and scores['scores']['real']['mixed'] < 12


def test_outcome_needs_noise_as_well_as_chance():
    yes, no = {'clear': True}, {'clear': False}
    assert read_outcome(dict(mixed_minus_chance=yes, mixed_minus_noise=no, within_minus_chance=yes, within_minus_noise=yes)) == FIRST
    assert read_outcome(dict(mixed_minus_chance=yes, mixed_minus_noise=yes, within_minus_chance=yes, within_minus_noise=no)) == SECOND
    assert read_outcome(dict(mixed_minus_chance=yes, mixed_minus_noise=yes, within_minus_chance=yes, within_minus_noise=yes)) == THIRD


def test_both_file_formats_parse(tmp_path):
    first = tmp_path/'a.txt'
    first.write_text('start####\nPredicted: one two\nTrue: three four\nend####\n\n\nstart####\nPredicted: five\nTrue: six\nend####\n', encoding='utf-8')
    second = tmp_path/'b.txt'
    second.write_text('target string: three four\npredicted string with tf: x\npredicted string: one two\n####\n\n\n', encoding='utf-8')
    assert read_outputs(first) == [{'predicted': 'one two', 'target': 'three four'}, {'predicted': 'five', 'target': 'six'}]
    assert read_outputs(second) == [{'target': 'three four', 'teacher_forced': 'x', 'predicted': 'one two'}]


def test_corpus_labels_match_exact_and_near_copies_only():
    local = ['He was born in Vienna in 1900 and studied law.']
    labels = label_corpus(['he was born in Vienna in 1900, and studied law', 'He was born in Vienna in 1900 and studied', 'A dull film with no plot.'], local)
    assert list(labels) == ['biography', 'biography', 'movie_review']
