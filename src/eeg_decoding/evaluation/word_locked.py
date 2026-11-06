"""Predict word meaning from word-locked EEG, then ask whether that tells held-out sentences apart within a block."""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix
from .averaged_eeg import _interval, resample_gap
from .block_holdout import position_features
from .nway import expected_top1, score_counts
from .retrieval import normalize

PRIMARY = 10
WAYS = (2, 4, 10, 24)
FIRST = 'Word-locked EEG carries sentence information beyond the controls'
SECOND = 'The gain does not exceed gaze, word properties or position'
THIRD = 'No sentence signal found with word-locked features'


@dataclass
class WordData:
    """Word rows over a sentence Layout. Tokens are targets only and never enter an EEG arm."""
    layout: object
    reader: np.ndarray
    sentence: np.ndarray
    slot: np.ndarray
    durations: np.ndarray
    fixations: np.ndarray
    tokens: list
    type_vectors: np.ndarray
    type_lengths: np.ndarray

    def __post_init__(self):
        self.row_type = np.array([self.tokens[s][p] for s, p in zip(self.sentence, self.slot)], dtype=np.int64)
        self.candidates = np.array([self.type_vectors[t].mean(0) for t in self.tokens])
        self.sentence_position = np.array([self.layout.place[r][i] for r, i in self.layout.identities])
        self.sentence_release = np.array([r for r, _ in self.layout.identities])

    def sentences_outside(self, release, low, high):
        buffer = self.layout.buffer
        return ~((self.sentence_release == release) & (self.sentence_position >= low - buffer) & (self.sentence_position <= high + buffer))

    def train_sentences(self, index):
        fold = self.layout.folds[index]
        return self.sentences_outside(fold.release, fold.low, fold.high)


def scale_rows(features, readers, fit):
    """Z-score per reader from fitting rows only. Missing values become 0 afterwards."""
    features = np.asarray(features, dtype=np.float64)
    output = np.zeros_like(features)
    for reader in np.unique(readers):
        rows = readers == reader
        chosen = rows & fit
        if not chosen.any():
            raise ValueError('A reader has no fitting rows')
        part = features[chosen]
        mean, spread = np.nanmean(part, 0), np.nanstd(part, 0)
        output[rows] = (features[rows] - np.nan_to_num(mean)) / np.nan_to_num(spread, nan=1.).clip(1e-6)
    return np.nan_to_num(output)


class GramSolver:
    """Ridge path through the feature covariance, for many more rows than features."""
    def __init__(self, x):
        self.mean = x.mean(0)
        scale = x.std(0)
        self.scale = np.where(scale > 0, scale, 1.)
        self.z = (x - self.mean) / self.scale
        values, self.v = np.linalg.eigh(self.z.T @ self.z)
        keep = values > max(values.max(), 1.) * 1e-12
        self.values, self.v = values[keep], self.v[:, keep]

    def project(self, x):
        return ((x - self.mean) / self.scale) @ self.v

    def path(self, y):
        mean = y.mean(0)
        return mean, self.v.T @ (self.z.T @ (y - mean))

    def predict(self, projected, fitted, alpha):
        mean, moment = fitted
        return mean + (projected / (self.values + alpha)) @ moment


def centered_scores(predicted, candidates):
    """Cosine after removing each pool's mean, so a candidate close to the average sentence cannot win for every query."""
    return _unit(predicted - predicted.mean(0)) @ _unit(candidates - candidates.mean(0)).T


def _unit(x):
    """Unit rows in double precision. A row with no length left scores 0 everywhere, so it ties and counts as a miss."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Embeddings must be finite matrices')
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > max(float(norms.max()), 1e-300) * 1e-9)


def zero_rows(scored, averaged):
    """How many sentences have no prediction left after centring within their block."""
    return int(sum((np.abs(_unit(averaged[b] - averaged[b].mean(0))).sum(1) == 0).sum() for b in scored.blocks))


def within_score(predicted, candidates, way=PRIMARY):
    below = score_counts(centered_scores(predicted, candidates), np.arange(len(candidates)))[0]
    return expected_top1(below, len(candidates), way)


def sentence_means(predicted, sentence, slot, weights=None):
    """Mean across readers per word slot, then mean across slots per sentence."""
    unique_sentence, sentence_index = np.unique(sentence, return_inverse=True)
    pairs = sentence_index.astype(np.int64) * (int(slot.max()) + 1) + slot
    unique_pair, pair_index = np.unique(pairs, return_inverse=True)
    weights = np.ones(len(pair_index)) if weights is None else weights
    rows = np.arange(len(pair_index))
    total = csr_matrix((weights, (pair_index, rows)), shape=(len(unique_pair), len(rows))) @ predicted
    count = np.bincount(pair_index, weights, len(unique_pair))
    used = count > 0
    owner = (unique_pair // (int(slot.max()) + 1))[used]
    slot_mean = total[used] / count[used, None]
    grouped = csr_matrix((np.ones(len(owner)), (owner, np.arange(len(owner)))), shape=(len(unique_sentence), len(owner))) @ slot_mean
    return grouped / np.bincount(owner, minlength=len(unique_sentence))[:, None], unique_sentence


class Table:
    """Word targets per sentence's slots, and the candidate vector each sentence is scored against."""
    def __init__(self, data, tokens=None):
        self.tokens = tokens if tokens is not None else data.tokens
        self.vectors = data.type_vectors
        self.candidates = np.array([self.vectors[t].mean(0) for t in self.tokens]) if tokens is not None else data.candidates

    def targets(self, sentence, slot):
        return np.array([self.vectors[self.tokens[s][p % len(self.tokens[s])]] for s, p in zip(sentence, slot)], dtype=np.float64)


def shuffled_table(data, index, seed):
    """Give each training sentence another training sentence's words in the same release, by slot, wrapping when shorter."""
    train = data.train_sentences(index)
    rows = np.zeros(len(data.tokens), dtype=bool)
    rows[np.unique(data.sentence)] = True
    tokens = list(data.tokens)
    for number, release in enumerate(data.layout.releases):
        members = np.flatnonzero(train & rows & (data.sentence_release == release))
        order = np.random.default_rng([seed, index, number]).permutation(members)
        for a, b in zip(members, order):
            tokens[a] = data.tokens[b]
    return Table(data, tokens)


def fit_fold(data, index, x, tables, alphas, seed, scale=True, fit_readers=None, score_readers=None):
    every = np.ones(len(data.reader), dtype=bool)
    fit_readers = every if fit_readers is None else fit_readers
    score_readers = every if score_readers is None else score_readers
    layout = data.layout
    train_sentence = data.train_sentences(index)
    inner = layout.inner_blocks(index, seed)
    inner_sentence = train_sentence.copy()
    for block in inner:
        inner_sentence &= data.sentences_outside(block.release, block.low, block.high)
    inner_train = inner_sentence[data.sentence]
    inner_x = scale_rows(x, data.reader, inner_train) if scale else np.asarray(x, dtype=np.float64)
    fit_rows = inner_train & fit_readers
    solver = GramSolver(inner_x[fit_rows])
    fitted = [solver.path(t.targets(data.sentence[fit_rows], data.slot[fit_rows])) for t in tables]
    inner_rows = [np.flatnonzero(np.isin(data.sentence, block.sentences) & fit_readers) for block in inner]
    inner_projected = [solver.project(inner_x[rows]) for rows in inner_rows]
    best = [(-1., None)] * len(tables)
    for alpha in alphas:
        for t, table in enumerate(tables):
            values = []
            for rows, projected in zip(inner_rows, inner_projected):
                predicted, unique = sentence_means(solver.predict(projected, fitted[t], alpha), data.sentence[rows], data.slot[rows])
                values.append(within_score(predicted, table.candidates[unique]))
            value = float(np.concatenate(values).mean())
            if value > best[t][0]:
                best[t] = (value, alpha)
    train = train_sentence[data.sentence]
    outer_x = scale_rows(x, data.reader, train) if scale else np.asarray(x, dtype=np.float64)
    fit_rows = train & fit_readers
    solver = GramSolver(outer_x[fit_rows])
    rows = np.flatnonzero(np.isin(data.sentence, layout.folds[index].sentences) & score_readers)
    projected = solver.project(outer_x[rows])
    predictions = [solver.predict(projected, solver.path(t.targets(data.sentence[fit_rows], data.slot[fit_rows])), best[n][1]).astype(np.float32)
                   for n, t in enumerate(tables)]
    return rows, predictions, best


def run_arm(data, options, tables_for, alphas, seed, **settings):
    """Fit every fold. `options(index)` lists candidate feature matrices; each table keeps its best."""
    collected, chosen, all_rows = [], [], []
    for index in range(len(data.layout.folds)):
        tables = tables_for(index)
        best = None
        for number, x in enumerate(options(index)):
            rows, predictions, scores = fit_fold(data, index, x, tables, alphas, seed, **settings)
            if best is None:
                best = [(s[0], s[1], number, p) for s, p in zip(scores, predictions)]
            else:
                best = [(s[0], s[1], number, p) if s[0] > b[0] else b for s, p, b in zip(scores, predictions, best)]
        all_rows.append(rows)
        collected.append([b[3] for b in best])
        chosen.append([{'inner_score': float(b[0]), 'alpha': float(b[1]), 'option': b[2]} for b in best])
    return np.concatenate(all_rows), [np.concatenate([fold[t] for fold in collected]) for t in range(len(collected[0]))], chosen


def noise_options(data, x, seed):
    """Keep each observation's mean word vector, and replace word-specific EEG with noise at the reader's within-observation training spread."""
    x = np.asarray(x, dtype=np.float32)
    keys = data.reader.astype(np.int64) * (int(data.sentence.max()) + 1) + data.sentence
    unique, inverse = np.unique(keys, return_inverse=True)
    present = np.isfinite(x)
    totals = np.zeros((len(unique), x.shape[1]))
    counts = np.zeros((len(unique), x.shape[1]))
    np.add.at(totals, inverse, np.nan_to_num(x))
    np.add.at(counts, inverse, present)
    means = (totals / np.maximum(counts, 1)).astype(np.float32)[inverse]
    noise = np.random.default_rng(seed).standard_normal(x.shape, dtype=np.float32)

    def options(index):
        train = data.train_sentences(index)[data.sentence]
        spread = np.ones(x.shape, dtype=np.float32)
        for reader in np.unique(data.reader):
            rows = data.reader == reader
            spread[rows] = np.nan_to_num(np.nanstd((x - means)[rows & train], 0), nan=1.)
        output = means + noise * spread
        output[~present] = np.nan
        return [output]
    return options


def gaze_options(data):
    """Word and reading properties with no EEG. Frequency comes from each fold's training rows."""
    lengths = data.type_lengths[data.row_type].astype(np.float64)
    words = np.array([len(data.tokens[s]) for s in data.sentence], dtype=np.float64)
    flag = (data.sentence_release[data.sentence] == data.layout.releases[0]).astype(np.float64)

    def options(index):
        train = data.train_sentences(index)[data.sentence]
        counts = np.bincount(data.row_type[train], minlength=len(data.type_vectors))
        frequency = np.log1p(counts[data.row_type])
        return [np.c_[lengths, frequency, data.durations, data.fixations, data.slot, words, flag]]
    return options


class WordScored:
    def __init__(self, data, rows, sentence_text=None):
        self.data, self.rows = data, rows
        layout = data.layout
        self.sentence, self.slot, self.reader = data.sentence[rows], data.slot[rows], data.reader[rows]
        self.order = layout.scored
        where = {s: i for i, s in enumerate(self.order)}
        self.position = np.array([where[s] for s in self.sentence])
        self.blocks, start = [], 0
        for fold in layout.folds:
            self.blocks.append(np.arange(start, start + len(fold.sentences)))
            start += len(fold.sentences)
        self.block_of = np.zeros(len(self.order), dtype=int)
        for i, b in enumerate(self.blocks):
            self.block_of[b] = i
        self.candidates = data.candidates[self.order]
        self.sentence_text = None if sentence_text is None else sentence_text[self.order]

    def average(self, predictions, weights=None):
        if weights is not None:
            empty = np.bincount(self.position, weights, len(self.order)) <= 0
            if empty.any():
                # A resample can miss every reader of a sentence; that sentence keeps all its readers.
                weights = np.where(empty[self.position], 1., weights)
        means, unique = sentence_means(predictions, self.position, self.slot, weights)
        if len(unique) != len(self.order):
            raise ValueError('A scored sentence has no word rows')
        return means

    def within(self, averaged, candidates=None, way=PRIMARY):
        candidates = self.candidates if candidates is None else candidates
        output = np.zeros(len(self.order))
        for block in self.blocks:
            output[block] = within_score(averaged[block], candidates[block], way)
        return output

    def across(self, averaged, ways=WAYS):
        output = {}
        releases = self.data.sentence_release[self.order]
        for release in self.data.layout.releases:
            members = np.flatnonzero(releases == release)
            below = score_counts(centered_scores(averaged[members], self.candidates[members]), np.arange(len(members)))[0]
            output[release] = {f'{w}_way': float(expected_top1(below, len(members), w).mean()) for w in ways}
        return output

    def by_release(self, values):
        releases = self.data.sentence_release[self.order]
        return {release: float(values[releases == release].mean()) for release in self.data.layout.releases}

    def covariates(self):
        """List position, word count, mean reading time and fixated slots; no EEG amplitudes."""
        data = self.data
        output = np.zeros((len(self.order), 4))
        output[:, 0] = data.sentence_position[self.order]
        output[:, 1] = [len(data.tokens[s]) for s in self.order]
        output[:, 2] = np.bincount(self.position, data.durations[self.rows], len(self.order)) / np.bincount(self.position, minlength=len(self.order))
        for i in range(len(self.order)):
            output[i, 3] = len(np.unique(self.slot[self.position == i]))
        return output

    def partial(self, averaged, covariates):
        """Within each block, remove what position, its square, word count, reading time and fixated slots explain linearly."""
        output = np.array(averaged, dtype=np.float64)
        for block in self.blocks:
            c = covariates[block]
            c = (c - c.mean(0)) / c.std(0).clip(1e-6)
            design = np.c_[np.ones(len(block)), c, c[:, 0] ** 2]
            coefficient, *_ = np.linalg.lstsq(design, output[block], rcond=None)
            output[block] -= design @ coefficient
        return output

    def curve(self, predictions, counts, draws, seed):
        output = {}
        readers_of = {i: np.unique(self.reader[self.position == i]) for i in range(len(self.order))}
        keys = self.position.astype(np.int64) * (int(self.reader.max()) + 1) + self.reader
        for count in counts:
            if count == 'all':
                output['all'] = float(self.within(self.average(predictions)).mean())
                continue
            values = []
            for draw in range(draws):
                rng = np.random.default_rng([seed, count, draw])
                chosen = [i * (int(self.reader.max()) + 1) + r for i, members in readers_of.items()
                          for r in rng.choice(members, min(count, len(members)), replace=False)]
                values.append(self.within(self.average(predictions, np.isin(keys, chosen).astype(float))).mean())
            output[str(count)] = float(np.mean(values))
        return output

    def word_test(self, predictions, way=PRIMARY):
        """Expected N-way among each block's unique word types, one equal-weight score per type."""
        types = self.data.row_type[self.rows]
        vectors = self.data.type_vectors
        scores = []
        for block in self.blocks:
            rows = np.isin(self.position, block)
            unique, inverse = np.unique(types[rows], return_inverse=True)
            if len(unique) < way:
                continue
            total = np.zeros((len(unique), predictions.shape[1]))
            np.add.at(total, inverse, predictions[rows])
            mean = total / np.bincount(inverse)[:, None]
            scores.append(within_score(mean, vectors[unique], way))
        return float(np.concatenate(scores).mean())


def resample_readers(scored, real, comparators, iterations, seed):
    rng = np.random.default_rng(seed)
    layout = scored.data.layout
    groups = [[r for r in range(len(layout.readers)) if layout.reader_release[r] == release] for release in layout.releases]
    gaps = {name: np.zeros(iterations) for name in comparators}
    for i in range(iterations):
        weight = np.zeros(len(layout.readers))
        for group in groups:
            weight[group] = np.bincount(rng.integers(len(group), size=len(group)), minlength=len(group))
        row_weight = weight[scored.reader]
        value = scored.within(scored.average(real, row_weight)).mean()
        for name, stack in comparators.items():
            gaps[name][i] = value - np.mean([scored.within(scored.average(p, row_weight)).mean() for p in stack])
    return {name: _interval(values) for name, values in gaps.items()}


def read_outcome(gaps, lower_bounds, minimum=.02):
    """Frozen reading. A gap is clear at the minimum with every available lower bound above 0."""
    clear = lambda names: all(gaps[n] >= minimum and all(v > 0 for v in lower_bounds[n].values()) for n in names)
    beats = clear(('gaussian', 'class_shuffled'))
    label = THIRD if not beats else FIRST if clear(('gaze_word', 'position', 'partial')) else SECOND
    return {'label': label, 'beats_nulls': beats, 'gaps': gaps, 'lower_bounds': lower_bounds, 'minimum_gap': minimum}


def half_readers(data, seed):
    layout = data.layout
    fit = np.zeros(len(layout.readers), dtype=bool)
    rng = np.random.default_rng(seed)
    for release in layout.releases:
        group = [r for r in range(len(layout.readers)) if layout.reader_release[r] == release]
        fit[rng.permutation(group)[:len(group) // 2]] = True
    return fit[data.reader], ~fit[data.reader]


def seen_type_share(data):
    """Share of held-out word rows whose type also occurs in that fold's training rows."""
    seen, total = 0, 0
    for index, fold in enumerate(data.layout.folds):
        train = data.train_sentences(index)[data.sentence]
        known = np.isin(data.row_type, np.unique(data.row_type[train]))
        held = np.isin(data.sentence, fold.sentences)
        seen += int((known & held).sum())
        total += int(held.sum())
    return seen / total


def run_test(data, bands, waves, sentence_text, position_terms, plan, status=print, iterations=10000, reader_iterations=2000):
    """Every arm and reading in the frozen plan."""
    seed = plan['seed']
    layout = data.layout
    alphas, shuffles = plan['model']['alphas'], plan['arms_shuffles']
    counts, draws = plan['sentence_prediction']['reader_counts_for_curve'], plan['sentence_prediction']['subset_draws_count']
    base = Table(data)
    true = lambda index: [base]
    with_shuffles = lambda index: [base] + [shuffled_table(data, index, seed + s) for s in range(shuffles)]
    features = lambda matrix: (lambda index: [matrix])
    status('FIT real and shuffled pairs')
    rows, predicted, chosen = run_arm(data, features(bands), with_shuffles, alphas, seed)
    real, shuffled = predicted[0], predicted[1:]
    scored = WordScored(data, rows, sentence_text)
    selection = {'real': [c[0] for c in chosen]}
    status('FIT matched noise')
    noise = noise_options(data, bands, seed)
    _, predicted, chosen = run_arm(data, noise, true, alphas, seed)
    gaussian = predicted[0]
    selection['gaussian'] = [c[0] for c in chosen]
    status('FIT gaze and word properties')
    _, predicted, chosen = run_arm(data, gaze_options(data), true, alphas, seed, scale=False)
    gaze = predicted[0]
    selection['gaze_word'] = [c[0] for c in chosen]
    status('FIT list position')
    ranges = {release: (ids[0], ids[-1]) for release, ids in layout.order.items()}

    class Row:
        def __init__(self, identity):
            self.dataset, self.sentence_id = identity

    by_sentence = [position_features([Row(i) for i in layout.identities], terms, ranges) for terms in position_terms]
    place = [matrix[data.sentence] for matrix in by_sentence]
    _, predicted, chosen = run_arm(data, lambda index: place, true, alphas, seed, scale=False)
    position = predicted[0]
    selection['position'] = [c[0] for c in chosen]

    arms = {'real': real, 'gaussian': gaussian, 'gaze_word': gaze, 'position': position}
    values = {name: scored.within(scored.average(p)) for name, p in arms.items()}
    values['class_shuffled'] = np.mean([scored.within(scored.average(p)) for p in shuffled], axis=0)
    covariates = scored.covariates()
    values['real_partial'] = scored.within(scored.partial(scored.average(real), covariates))
    values['gaussian_partial'] = scored.within(scored.partial(scored.average(gaussian), covariates))
    primary = {name: float(v.mean()) for name, v in values.items()}
    status(f'Primary 10-way within block: {primary}')
    unscorable = {name: zero_rows(scored, scored.partial(scored.average(p), covariates)) for name, p in (('real', real), ('gaussian', gaussian))}
    gaps = {name: primary['real'] - primary[name] for name in ('gaussian', 'class_shuffled', 'gaze_word', 'position')}
    gaps['partial'] = primary['real_partial'] - primary['gaussian_partial']
    intervals = {name: resample_gap(values['real'] - values[name], scored.block_of, iterations, seed) for name in gaps if name != 'partial'}
    intervals['partial'] = resample_gap(values['real_partial'] - values['gaussian_partial'], scored.block_of, iterations, seed)
    status('RESAMPLE readers')
    readers = resample_readers(scored, real, {'gaussian': [gaussian], 'class_shuffled': shuffled}, reader_iterations, seed)
    lower = {name: {'sentence': intervals[name]['sentence_interval'][0], 'block': intervals[name]['block_interval'][0]} for name in gaps}
    for name in readers:
        lower[name]['reader'] = readers[name][0]
    decision = read_outcome(gaps, lower, plan['decision_rule']['minimum_gap_points'] / 100)
    decision.update(primary=primary, intervals=intervals, reader_intervals=readers, partial_sentences_with_no_prediction_left=unscorable)

    status('SECONDARY reader counts, pools and word test')
    secondary = {'by_release': {name: scored.by_release(v) for name, v in values.items()},
                 'by_block': {name: [float(v[b].mean()) for b in scored.blocks] for name, v in values.items()},
                 'all_sentence_pool': {name: scored.across(scored.average(p)) for name, p in arms.items()},
                 'reader_curve': {name: scored.curve(arms[name], counts, draws, seed) for name in ('real', 'gaussian')},
                 'direct_word_test': {name: scored.word_test(arms[name]) for name in ('real', 'gaussian', 'gaze_word')},
                 'seen_type_share': seen_type_share(data)}
    if scored.sentence_text is not None:
        secondary['sentence_embedding_candidates'] = {name: float(scored.within(scored.average(p), scored.sentence_text).mean()) for name, p in arms.items()}
    status('SECONDARY waveform input')
    _, predicted, _ = run_arm(data, features(waves), true, alphas, seed)
    secondary['waveform'] = {'real': float(scored.within(scored.average(predicted[0])).mean())}
    _, predicted, _ = run_arm(data, noise_options(data, waves, seed), true, alphas, seed)
    secondary['waveform']['gaussian'] = float(scored.within(scored.average(predicted[0])).mean())
    status('SECONDARY new readers')
    fit_readers, score_readers = half_readers(data, seed)
    new_rows, predicted, _ = run_arm(data, features(bands), with_shuffles, alphas, seed, fit_readers=fit_readers, score_readers=score_readers)
    new = WordScored(data, new_rows)
    new_real = new.within(new.average(predicted[0]))
    new_shuffled = np.mean([new.within(new.average(p)) for p in predicted[1:]], axis=0)
    _, predicted, _ = run_arm(data, noise, true, alphas, seed, fit_readers=fit_readers, score_readers=score_readers)
    new_gaussian = new.within(new.average(predicted[0]))
    secondary['new_readers'] = {'real': float(new_real.mean()), 'gaussian': float(new_gaussian.mean()), 'class_shuffled': float(new_shuffled.mean()),
                                'real_minus_gaussian': resample_gap(new_real - new_gaussian, new.block_of, iterations, seed),
                                'real_minus_class_shuffled': resample_gap(new_real - new_shuffled, new.block_of, iterations, seed),
                                'fit_readers': int(len(set(data.reader[fit_readers]))), 'scored_readers': int(len(set(data.reader[score_readers])))}
    return {'decision': decision, 'secondary': secondary, 'selection': selection, 'values': values}


__all__ = ['FIRST', 'centered_scores', 'zero_rows', 'GramSolver', 'SECOND', 'THIRD', 'Table', 'WordData', 'WordScored', 'fit_fold', 'gaze_options', 'half_readers', 'noise_options',
           'read_outcome', 'resample_readers', 'run_arm', 'run_test', 'scale_rows', 'seen_type_share', 'sentence_means', 'shuffled_table']
