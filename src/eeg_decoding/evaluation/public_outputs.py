"""Score other authors' published EEG-to-text outputs within and across corpora. No EEG is read and nothing is fitted."""
from math import comb
import re
import numpy as np
from .retrieval import normalize

PRIMARY = 10
WAYS = (2, 4, 10, 24)
FIRST = 'No retrieval above chance or noise in these outputs'
SECOND = 'The mixed score comes mostly from telling the corpora apart'
THIRD = 'The score holds within a corpus'
FIELDS = {'Predicted:': 'predicted', 'True:': 'target', 'target string:': 'target', 'predicted string with tf:': 'teacher_forced', 'predicted string:': 'predicted'}


def squash(sentence):
    return re.sub(r'[^a-z0-9]+', '', sentence.lower())


def read_outputs(path):
    """One dict per test row from either repository's text format."""
    rows, row = [], {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            for label, name in FIELDS.items():
                if line.startswith(label):
                    if name in row:
                        raise ValueError(f'Repeated field in one record: {label}')
                    row[name] = line[len(label):].strip()
                    break
            else:
                if line.startswith(('end#', '#')) and row:
                    rows.append(row)
                    row = {}
    if row:
        rows.append(row)
    if any('target' not in r or 'predicted' not in r for r in rows):
        raise ValueError('A record lacks a target or a prediction')
    return rows


def tokens(sentence):
    return frozenset(re.findall(r'[a-z0-9]+', sentence.lower()))


def label_corpus(targets, biography_texts, jaccard=.7):
    """Biography when a target matches a local normal-reading sentence; movie review otherwise, by elimination."""
    exact = {squash(t) for t in biography_texts}
    bags = [tokens(t) for t in biography_texts]
    labels = []
    for target in targets:
        bag = tokens(target)
        near = squash(target) in exact or any(len(bag & other) >= jaccard * len(bag | other) for other in bags)
        labels.append('biography' if near else 'movie_review')
    return np.array(labels)


def expected_scores(predicted, own, bank, pool, way):
    """Exact expected top-1 per row, with way-1 distractors drawn from the other sentences in each row's pool. Ties are misses."""
    scores = predicted @ bank.T
    beaten = (scores[np.arange(len(own)), own][:, None] > scores) & pool[own]
    others = pool[own].sum(1) - 1
    return np.array([comb(int(b), way - 1) / comb(int(o), way - 1) if o >= way - 1 else np.nan for b, o in zip(beaten.sum(1), others)])


def by_sentence(values, own, count):
    return np.array([np.nanmean(values[own == s]) if (own == s).any() else np.nan for s in range(count)])


def sentence_scores(predicted, own, bank, corpus, way=PRIMARY):
    """Per-sentence expected score against all sentences, and against the sentence's own corpus only."""
    same = corpus[:, None] == corpus[None, :]
    np_all = np.ones_like(same)
    return (by_sentence(expected_scores(predicted, own, bank, np_all, way), own, len(bank)),
            by_sentence(expected_scores(predicted, own, bank, same, way), own, len(bank)))


def label_only(own, bank, corpus):
    """A predictor with no EEG that knows only the query's corpus: the mean text of the other test sentences in it."""
    output = np.zeros((len(own), bank.shape[1]))
    for row, sentence in enumerate(own):
        members = (corpus == corpus[sentence]) & (np.arange(len(bank)) != sentence)
        output[row] = bank[members].mean(0)
    return normalize(output)


def interval(values, draws):
    means = values[draws].mean(1)
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def gap(values, draws, minimum):
    """Mean of per-sentence differences in points, with a sentence bootstrap."""
    low, high = interval(values, draws)
    mean = float(values.mean())
    return {'points': mean, 'sentence_interval_95': [low, high], 'clear': bool(mean >= minimum and low > 0)}


def read_outcome(checks):
    mixed = checks['mixed_minus_chance']['clear'] and checks['mixed_minus_noise']['clear']
    within = checks['within_minus_chance']['clear'] and checks['within_minus_noise']['clear']
    return FIRST if not mixed else THIRD if within else SECOND


def pair_split(predicted, own, bank, corpus):
    """2-way against same-corpus and different-corpus distractors, one equal-weight value per sentence."""
    scores = predicted @ bank.T
    wins = scores[np.arange(len(own)), own][:, None] > scores
    count = len(bank)
    matrix = np.array([wins[own == s].mean(0) for s in range(count)])
    same = corpus[:, None] == corpus[None, :]
    np.fill_diagonal(same, False)
    return float(100 * matrix[same].mean()), float(100 * matrix[corpus[:, None] != corpus[None, :]].mean())


def run_check(vectors, own, bank, corpus, plan):
    """`vectors` maps arm names to unit rows; it must hold 'real' and 'noise'. Returns the decision and secondary scores."""
    rule = plan['decision_rule']
    minimum = rule['minimum_gap_points']
    rng = np.random.default_rng(plan['seed'])
    count = len(bank)
    draws = rng.integers(count, size=(rule.get('resamples', 10000), count))
    vectors = dict(vectors, corpus_label_only=label_only(own, bank, corpus))
    primary = {name: [100 * s for s in sentence_scores(v, own, bank, corpus)] for name, v in vectors.items()}
    mixed, within = primary['real']
    noise_mixed, noise_within = primary['noise']
    chance = 100 / PRIMARY
    checks = {'mixed_minus_chance': gap(mixed - chance, draws, minimum), 'mixed_minus_noise': gap(mixed - noise_mixed, draws, minimum),
              'within_minus_chance': gap(within - chance, draws, minimum), 'within_minus_noise': gap(within - noise_within, draws, minimum),
              'mixed_minus_within': gap(mixed - within, draws, minimum)}
    groups = sorted(set(corpus))
    scores = {name: dict(mixed=float(m.mean()), within_corpus=float(w.mean()), **{g: float(w[corpus == g].mean()) for g in groups}) for name, (m, w) in primary.items()}
    decision = {'label': read_outcome(checks), 'primary_way': PRIMARY, 'chance': chance, 'scores': scores, 'checks': checks,
                'sentences': count, 'sentences_by_corpus': {g: int((corpus == g).sum()) for g in groups}, 'rows': len(own)}
    secondary = {'n_way': {}, 'two_way_split': {}}
    for name, v in vectors.items():
        entry = {}
        for way in WAYS:
            m, w = sentence_scores(v, own, bank, corpus, way)
            entry[f'{way}_way'] = dict(mixed=float(100 * m.mean()), within_corpus=float(100 * np.nanmean(w)), **{g: float(100 * np.nanmean(w[corpus == g])) for g in groups})
        secondary['n_way'][name] = entry
        same, different = pair_split(v, own, bank, corpus)
        secondary['two_way_split'][name] = {'same_corpus': same, 'different_corpus': different}
    return {'decision': decision, 'secondary': secondary, 'values': {name: np.stack(pair) for name, pair in primary.items()}}
