"""Average EEG across readers and ask whether it tells held-out sentences apart within a list block."""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import svd
from scipy.sparse import csr_matrix
from .nway import expected_top1, score_counts
from .retrieval import normalize
from .sentence_holdout import TOKEN, sentence_groups, word_count
from .spectral_pilot import count_basis

PRIMARY = 10
WAYS = (2, 4, 10, 24)
FIRST = 'Averaged EEG carries sentence information beyond the controls'
SECOND = 'The gain does not exceed timing or position'
THIRD = 'No sentence signal found with these features'


@dataclass
class Fold:
    release: str
    low: int
    high: int
    sentences: np.ndarray


class Layout:
    """Usable rows and held-out blocks. Protected rows never enter this object."""
    def __init__(self, records, zone, v1_test, forbidden, block=25, buffer=10, minimum=10):
        self.block, self.buffer, self.minimum = block, buffer, minimum
        self.zone, self.forbidden = zone, forbidden
        self.order = {}
        for release in sorted({r.dataset for r in records}):
            self.order[release] = sorted({int(r.sentence_id) for r in records if r.dataset == release})
        self.place = {release: {i: p for p, i in enumerate(ids)} for release, ids in self.order.items()}
        self.rows = [r for r in records if int(r.sentence_id) not in zone.get(r.dataset, ()) and (r.dataset, int(r.sentence_id)) not in v1_test]
        self.identities = sorted({(r.dataset, int(r.sentence_id)) for r in self.rows})
        lookup = {identity: i for i, identity in enumerate(self.identities)}
        self.lookup = lookup
        self.releases = sorted(self.order)
        self.readers = sorted({r.subject_key for r in self.rows})
        readers = {name: i for i, name in enumerate(self.readers)}
        self.reader = np.array([readers[r.subject_key] for r in self.rows])
        self.sentence = np.array([lookup[r.dataset, int(r.sentence_id)] for r in self.rows])
        self.release = np.array([r.dataset for r in self.rows])
        self.position = np.array([self.place[r.dataset][int(r.sentence_id)] for r in self.rows])
        self.key_of = {}
        for r in self.rows:
            self.key_of[lookup[r.dataset, int(r.sentence_id)]] = r.key
        self.keys = [self.key_of[i] for i in range(len(self.identities))]
        self.sentence_release = np.array([i[0] for i in self.identities])
        self.reader_release = {readers[r.subject_key]: r.dataset for r in self.rows}
        self.folds = self._tile()
        self.scored = np.concatenate([f.sentences for f in self.folds])

    def _scoreable(self, release, position):
        identity = self.order[release][position]
        return (release, identity) in self.lookup and identity not in self.forbidden.get(release, ())

    def _tile(self):
        folds = []
        for release in self.releases:
            runs, current = [], []
            for p, i in enumerate(self.order[release]):
                if i in self.zone.get(release, ()):
                    runs += [current] if current else []
                    current = []
                else:
                    current.append(p)
            runs += [current] if current else []
            for run in runs:
                for start in range(0, len(run) - self.block + 1, self.block):
                    span = run[start:start + self.block]
                    members = [self.lookup[release, self.order[release][p]] for p in span if self._scoreable(release, p)]
                    if len(members) >= self.minimum:
                        folds.append(Fold(release, span[0], span[-1], np.array(members)))
        return folds

    def outside(self, release, low, high):
        """Rows at least one buffer away from a block; the other release is always outside."""
        near = (self.release == release) & (self.position >= low - self.buffer) & (self.position <= high + self.buffer)
        return ~near

    def train_mask(self, index):
        fold = self.folds[index]
        return self.outside(fold.release, fold.low, fold.high)

    def inner_blocks(self, index, seed):
        """One seeded block per release, inside this fold's training data, for choosing settings."""
        fold = self.folds[index]
        output = []
        for number, release in enumerate(self.releases):
            closed = {self.place[release][i] for i in self.zone.get(release, ())}
            if release == fold.release:
                closed |= set(range(fold.low - self.buffer, fold.high + self.buffer + 1))
            count = len(self.order[release])
            starts = [s for s in range(count - self.block + 1) if not closed & set(range(s, s + self.block))]
            rng = np.random.default_rng([seed, index, number])
            for start in rng.permutation(starts):
                start = int(start)
                members = [self.lookup[release, self.order[release][p]] for p in range(start, start + self.block) if self._scoreable(release, p)]
                if len(members) >= self.minimum:
                    output.append(Fold(release, start, start + self.block - 1, np.array(members)))
                    break
            else:
                raise ValueError(f'No inner block for fold {index} in {release}')
        return output


def scale_by_reader(features, readers, fit):
    """Z-score each reader's features from that reader's fitting rows only."""
    features = np.asarray(features, dtype=np.float64)
    output = np.zeros_like(features)
    for reader in np.unique(readers):
        rows = readers == reader
        chosen = rows & fit
        if not chosen.any():
            raise ValueError('A reader has no fitting rows')
        output[rows] = (features[rows] - features[chosen].mean(0)) / features[chosen].std(0).clip(1e-6)
    return output


class Solver:
    """Ridge path with one scaling and SVD, reused for many alphas and target tables."""
    def __init__(self, x):
        self.mean = x.mean(0)
        scale = x.std(0)
        self.scale = np.where(scale > 0, scale, 1.)
        u, s, vt = svd((x - self.mean) / self.scale, full_matrices=False)
        keep = s > 1e-15
        self.u, self.s, self.v = u[:, keep], s[keep], vt[keep].T

    def project(self, x):
        return ((x - self.mean) / self.scale) @ self.v

    def path(self, y):
        mean = y.mean(0)
        return mean, self.u.T @ (y - mean)

    def predict(self, projected, fitted, alpha):
        mean, uty = fitted
        return mean + (projected * (self.s / (self.s ** 2 + alpha))) @ uty


def _means(x, sentences):
    unique, inverse = np.unique(sentences, return_inverse=True)
    total = np.zeros((len(unique), x.shape[1]))
    np.add.at(total, inverse, x)
    return total / np.bincount(inverse)[:, None], unique


def _fit_rows(x, sentence, fit, tables, fit_means):
    if fit_means:
        grouped, unique = _means(x[fit], sentence[fit])
        return Solver(grouped), [table[unique] for table in tables]
    return Solver(x[fit]), [table[sentence[fit]] for table in tables]


def within_score(predicted, targets, way=PRIMARY):
    """Expected N-way within one block, per sentence; ties are misses."""
    scores = normalize(predicted) @ normalize(targets).T
    below, _, _ = score_counts(scores, np.arange(len(targets)))
    return expected_top1(below, len(targets), way)


def fit_fold(layout, index, x, tables, alphas, seed, scale=True, fit_readers=None, score_readers=None, fit_means=False):
    """Choose alpha on an inner block per target table, then refit on the fold's training rows."""
    every = np.ones(len(layout.rows), dtype=bool)
    fit_readers = every if fit_readers is None else fit_readers
    score_readers = every if score_readers is None else score_readers
    train = layout.train_mask(index)
    inner = layout.inner_blocks(index, seed)
    inner_train = train.copy()
    for block in inner:
        inner_train &= layout.outside(block.release, block.low, block.high)
    inner_x = scale_by_reader(x, layout.reader, inner_train) if scale else np.asarray(x, dtype=np.float64)
    solver, targets = _fit_rows(inner_x, layout.sentence, inner_train & fit_readers, tables, fit_means)
    fitted = [solver.path(t) for t in targets]
    inner_rows = [(block, np.flatnonzero(np.isin(layout.sentence, block.sentences) & fit_readers)) for block in inner]
    inner_projected = [solver.project(inner_x[rows]) for _, rows in inner_rows]
    best = [(-1., None)] * len(tables)
    for alpha in alphas:
        for t, table in enumerate(tables):
            values = []
            for (block, rows), projected in zip(inner_rows, inner_projected):
                predicted, unique = _means(solver.predict(projected, fitted[t], alpha), layout.sentence[rows])
                values.append(within_score(predicted, table[unique]))
            value = float(np.concatenate(values).mean())
            if value > best[t][0]:
                best[t] = (value, alpha)
    outer_x = scale_by_reader(x, layout.reader, train) if scale else np.asarray(x, dtype=np.float64)
    solver, targets = _fit_rows(outer_x, layout.sentence, train & fit_readers, tables, fit_means)
    rows = np.flatnonzero(np.isin(layout.sentence, layout.folds[index].sentences) & score_readers)
    projected = solver.project(outer_x[rows])
    predictions = [solver.predict(projected, solver.path(y), best[t][1]).astype(np.float32) for t, y in enumerate(targets)]
    return rows, predictions, best


def run_arm(layout, options, tables_for, alphas, seed, **settings):
    """Fit every fold. `options(train)` lists candidate feature matrices; each table keeps its best."""
    collected, chosen = [], []
    all_rows = []
    for index in range(len(layout.folds)):
        tables = tables_for(index)
        best = None
        for number, x in enumerate(options(layout.train_mask(index))):
            rows, predictions, scores = fit_fold(layout, index, x, tables, alphas, seed, **settings)
            if best is None:
                best = [(s[0], s[1], number, p) for s, p in zip(scores, predictions)]
            else:
                best = [(s[0], s[1], number, p) if s[0] > b[0] else b for s, p, b in zip(scores, predictions, best)]
        all_rows.append(rows)
        collected.append([b[3] for b in best])
        chosen.append([{'inner_score': float(b[0]), 'alpha': float(b[1]), 'option': b[2]} for b in best])
    rows = np.concatenate(all_rows)
    return rows, [np.concatenate([fold[t] for fold in collected]) for t in range(len(collected[0]))], chosen


def shuffled_tables(layout, index, text, shuffles, seed):
    """Permute targets between this fold's training sentences within each release."""
    train = layout.train_mask(index)
    tables = []
    for shuffle in range(shuffles):
        table = text.copy()
        for number, release in enumerate(layout.releases):
            members = np.unique(layout.sentence[train & (layout.release == release)])
            table[members] = text[np.random.default_rng([seed, shuffle, index, number]).permutation(members)]
        tables.append(table)
    return tables


class Scored:
    """Turn row predictions into one prediction per sentence and score them."""
    def __init__(self, layout, rows, text):
        self.layout, self.text = layout, text
        where = {s: i for i, s in enumerate(layout.scored)}
        self.sentence = np.array([where[s] for s in layout.sentence[rows]])
        self.reader = layout.reader[rows]
        self.count = len(layout.scored)
        self.rows = np.arange(len(rows))
        self.blocks = []
        start = 0
        for fold in layout.folds:
            self.blocks.append(np.arange(start, start + len(fold.sentences)))
            start += len(fold.sentences)
        self.targets = [normalize(text[layout.scored[b]]) for b in self.blocks]
        self.block_of = np.zeros(self.count, dtype=int)
        for i, b in enumerate(self.blocks):
            self.block_of[b] = i

    def average(self, predictions, weights=None):
        weights = np.ones(len(self.rows)) if weights is None else weights
        if np.any(np.bincount(self.sentence, weights, self.count) <= 0):
            # A resample can miss every reader of a sentence; that sentence keeps all its readers.
            empty = np.bincount(self.sentence, weights, self.count) <= 0
            weights = np.where(empty[self.sentence], 1., weights)
        matrix = csr_matrix((weights, (self.sentence, self.rows)), shape=(self.count, len(self.rows)))
        return (matrix @ predictions) / np.bincount(self.sentence, weights, self.count)[:, None]

    def covariates(self, durations):
        """List position, word count and mean duration per scored sentence; none of these use EEG amplitudes or text meaning."""
        layout = self.layout
        rows = np.flatnonzero(np.isin(layout.sentence, layout.scored))
        where = {s: i for i, s in enumerate(layout.scored)}
        groups = np.array([where[s] for s in layout.sentence[rows]])
        output = np.zeros((self.count, 3))
        output[:, 2] = np.bincount(groups, np.asarray(durations, dtype=float)[rows], self.count) / np.bincount(groups, minlength=self.count)
        for i, s in enumerate(layout.scored):
            release, identity = layout.identities[s]
            output[i, 0] = layout.place[release][identity]
        counts = {s: word_count(r.text) for s, r in zip(layout.sentence, layout.rows)}
        output[:, 1] = [counts[s] for s in layout.scored]
        return output

    def partial(self, averaged, covariates):
        """Within each block, remove from the predictions what position, its square, word count and duration explain linearly."""
        output = np.array(averaged, dtype=np.float64)
        for block in self.blocks:
            c = covariates[block]
            c = (c - c.mean(0)) / c.std(0).clip(1e-6)
            design = np.c_[np.ones(len(block)), c, c[:, 0] ** 2]
            coefficient, *_ = np.linalg.lstsq(design, output[block], rcond=None)
            output[block] -= design @ coefficient
        return output

    def within(self, averaged, way=PRIMARY):
        output = np.zeros(self.count)
        for block, targets in zip(self.blocks, self.targets):
            output[block] = expected_top1(score_counts(normalize(averaged[block]) @ targets.T, np.arange(len(block)))[0], len(block), way)
        return output

    def across(self, averaged, ways=WAYS):
        """Expected N-way against all scored sentences in the release."""
        output = {}
        for release in self.layout.releases:
            members = np.flatnonzero(self.layout.sentence_release[self.layout.scored] == release)
            scores = normalize(averaged[members]) @ normalize(self.text[self.layout.scored[members]]).T
            below = score_counts(scores, np.arange(len(members)))[0]
            output[release] = {f'{w}_way': float(expected_top1(below, len(members), w).mean()) for w in ways}
        return output

    def by_release(self, values):
        return {release: float(values[self.layout.sentence_release[self.layout.scored] == release].mean()) for release in self.layout.releases}

    def curve(self, predictions, counts, draws, seed):
        """Score at each reader count; subsets differ per sentence and per draw."""
        output = {}
        for count in counts:
            if count == 'all':
                output['all'] = float(self.within(self.average(predictions)).mean())
                continue
            values = []
            for draw in range(draws):
                rng = np.random.default_rng([seed, count, draw])
                weights = np.zeros(len(self.rows))
                for sentence in range(self.count):
                    members = np.flatnonzero(self.sentence == sentence)
                    weights[rng.choice(members, min(count, len(members)), replace=False)] = 1.
                values.append(self.within(self.average(predictions, weights)).mean())
            output[str(count)] = float(np.mean(values))
        return output


def _interval(values):
    return np.percentile(values, [2.5, 97.5]).tolist()


def resample_gap(gap, block_of, iterations, seed):
    """95% percentile intervals of the mean gap, resampling sentences and blocks."""
    gap = np.asarray(gap, dtype=float)
    rng = np.random.default_rng(seed)
    by_sentence = gap[rng.integers(len(gap), size=(iterations, len(gap)))].mean(1)
    count = block_of.max() + 1
    totals, sizes = np.bincount(block_of, gap, count), np.bincount(block_of, minlength=count)
    picks = rng.integers(count, size=(iterations, count))
    return {'estimate': float(gap.mean()), 'sentence_interval': _interval(by_sentence), 'block_interval': _interval(totals[picks].sum(1) / sizes[picks].sum(1))}


def resample_readers(scored, real, comparators, iterations, seed):
    """Resample readers within release and recompute the average. Comparators are lists of predictions."""
    rng = np.random.default_rng(seed)
    layout = scored.layout
    readers = np.arange(len(layout.readers))
    groups = [[r for r in readers if layout.reader_release[r] == release] for release in layout.releases]
    gaps = {name: np.zeros(iterations) for name in comparators}
    for i in range(iterations):
        weight = np.zeros(len(readers))
        for group in groups:
            weight[group] = np.bincount(rng.integers(len(group), size=len(group)), minlength=len(group))
        row_weight = weight[scored.reader]
        value = scored.within(scored.average(real, row_weight)).mean()
        for name, stack in comparators.items():
            gaps[name][i] = value - np.mean([scored.within(scored.average(p, row_weight)).mean() for p in stack])
    return {name: _interval(values) for name, values in gaps.items()}


def read_outcome(gaps, lower_bounds, minimum=.02):
    """Frozen reading. Lower bounds are listed per null; every one has to clear zero."""
    clear = lambda names: all(gaps[n] >= minimum and all(v > 0 for v in lower_bounds[n].values()) for n in names)
    beats = clear(('gaussian', 'class_shuffled'))
    if not beats:
        label = THIRD
    elif clear(('timing', 'position', 'partial')):
        label = FIRST
    else:
        label = SECOND
    return {'label': label, 'beats_nulls': beats, 'gaps': gaps, 'lower_bounds': lower_bounds, 'minimum_gap': minimum}


def timing_options(layout, durations):
    counts = np.array([word_count(r.text) for r in layout.rows], dtype=float)

    def options(train):
        basis, _ = count_basis(counts[train], counts)
        return [np.c_[durations, counts, basis]]
    return options


def half_readers(layout, seed):
    """A seeded half of each release's readers fits; the other half is scored."""
    fit = np.zeros(len(layout.readers), dtype=bool)
    rng = np.random.default_rng(seed)
    for release in layout.releases:
        group = [r for r in range(len(layout.readers)) if layout.reader_release[r] == release]
        fit[rng.permutation(group)[:len(group) // 2]] = True
    return fit[layout.reader], ~fit[layout.reader]


def fold_audit(records, layout, text, closed, v1_test, jaccard=.7, containment=.8, cosine=.9):
    """Name every way a protected or held-out sentence could reach a fit or a score."""
    failures = []
    groups, _ = sentence_groups(records)
    group_of = {k: i for i, g in enumerate(groups) for k in g}
    usable = {(r.dataset, int(r.sentence_id)) for r in layout.rows}
    if usable & closed:
        failures.append('A v2 test-zone sentence is in the usable rows')
    if usable & v1_test:
        failures.append('A v1 test sentence is in the usable rows')
    wording = {r.key: set(TOKEN.findall(r.text.lower())) for r in records}
    worst = {'jaccard': 0., 'containment': 0., 'cosine': -1.}
    closest = 10 ** 9
    paths = np.array([r.signal_path for r in layout.rows])
    keys = np.array([r.key for r in layout.rows])
    for index, fold in enumerate(layout.folds):
        train = layout.train_mask(index)
        held = np.isin(layout.sentence, fold.sentences)
        if (train & held).any():
            failures.append(f'Fold {index}: a held-out row is in training')
        held_keys, train_keys = sorted(set(keys[held])), sorted(set(keys[train]))
        if set(held_keys) & set(train_keys):
            failures.append(f'Fold {index}: a sentence key is in training')
        if {group_of[k] for k in held_keys} & {group_of[k] for k in train_keys}:
            failures.append(f'Fold {index}: a sentence group is in training')
        if set(paths[held]) & set(paths[train]):
            failures.append(f'Fold {index}: a signal file is in training')
        same = train & (layout.release == fold.release)
        distance = int(min(np.abs(layout.position[same] - p).min() for p in layout.position[held]))
        closest = min(closest, distance)
        if distance <= layout.buffer:
            failures.append(f'Fold {index}: training is {distance} list positions away')
        for a in held_keys:
            for b in train_keys:
                shared = len(wording[a] & wording[b])
                worst['jaccard'] = max(worst['jaccard'], shared / max(len(wording[a] | wording[b]), 1))
                worst['containment'] = max(worst['containment'], shared / max(min(len(wording[a]), len(wording[b])), 1))
        worst['cosine'] = max(worst['cosine'], float((normalize(text.get(held_keys)) @ normalize(text.get(train_keys)).T).max()))
    if worst['jaccard'] >= jaccard:
        failures.append(f"Near-duplicate wording across a fold boundary: {worst['jaccard']:.3f}")
    if worst['containment'] >= containment:
        failures.append(f"One sentence nearly contains another across a fold boundary: {worst['containment']:.3f}")
    if worst['cosine'] >= cosine:
        failures.append(f"Held-out text nearly matches training: {worst['cosine']:.3f}")
    return {'failures': failures, 'worst': worst, 'min_list_distance_held_out_to_train': closest, 'blocks': len(layout.folds),
            'scored_sentences': int(len(layout.scored)), 'usable_observations': len(layout.rows), 'usable_sentences': len(layout.identities)}


def run_test(layout, text, features, durations, position_terms, plan, status=print, iterations=10000, reader_iterations=2000):
    """Every arm and reading in the frozen plan. `features` maps arm names to row-by-feature matrices."""
    from .block_holdout import position_features
    seed = plan['seed']
    alphas, shuffles = plan['model']['alphas'], plan['arms_shuffles']
    counts, draws = plan['averaging']['reader_counts_for_curve'], plan['averaging']['subset_draws_count']
    true = lambda index: [text]
    eeg = lambda matrix: (lambda train: [matrix])
    status('FIT real and shuffled pairs')
    rows, predicted, chosen = run_arm(layout, eeg(features['real']), lambda i: [text] + shuffled_tables(layout, i, text, shuffles, seed), alphas, seed)
    real, shuffled = predicted[0], predicted[1:]
    scored = Scored(layout, rows, text)
    selection = {'real': [c[0] for c in chosen]}
    status('FIT matched noise')
    _, predicted, chosen = run_arm(layout, eeg(features['gaussian']), true, alphas, seed)
    gaussian = predicted[0]
    selection['gaussian'] = [c[0] for c in chosen]
    status('FIT timing')
    _, predicted, chosen = run_arm(layout, timing_options(layout, durations), true, alphas, seed, scale=False)
    timing = predicted[0]
    selection['timing'] = [c[0] for c in chosen]
    status('FIT list position')
    ranges = {release: (ids[0], ids[-1]) for release, ids in layout.order.items()}
    place = [position_features(layout.rows, terms, ranges) for terms in position_terms]
    _, predicted, chosen = run_arm(layout, lambda train: place, true, alphas, seed, scale=False)
    position = predicted[0]
    selection['position'] = [c[0] for c in chosen]

    values = {name: scored.within(scored.average(p)) for name, p in (('real', real), ('gaussian', gaussian), ('timing', timing), ('position', position))}
    values['class_shuffled'] = np.mean([scored.within(scored.average(p)) for p in shuffled], axis=0)
    # A noisy readout of timing or position can outscore a clean one under cosine, so the gate also removes them from the predictions.
    covariates = scored.covariates(durations)
    values['real_partial'] = scored.within(scored.partial(scored.average(real), covariates))
    values['gaussian_partial'] = scored.within(scored.partial(scored.average(gaussian), covariates))
    primary = {name: float(v.mean()) for name, v in values.items()}
    status(f'Primary 10-way within block: {primary}')
    gaps = {name: primary['real'] - primary[name] for name in ('gaussian', 'class_shuffled', 'timing', 'position')}
    gaps['partial'] = primary['real_partial'] - primary['gaussian_partial']
    intervals = {name: resample_gap(values['real'] - values[name], scored.block_of, iterations, seed) for name in gaps if name != 'partial'}
    intervals['partial'] = resample_gap(values['real_partial'] - values['gaussian_partial'], scored.block_of, iterations, seed)
    status('RESAMPLE readers')
    readers = resample_readers(scored, real, {'gaussian': [gaussian], 'class_shuffled': shuffled}, reader_iterations, seed)
    lower = {name: {'sentence': intervals[name]['sentence_interval'][0], 'block': intervals[name]['block_interval'][0]} for name in gaps}
    for name in readers:
        lower[name]['reader'] = readers[name][0]
    decision = read_outcome(gaps, lower, plan['decision_rule']['minimum_gap_points'] / 100)
    decision['primary'] = primary
    decision['intervals'] = intervals
    decision['reader_intervals'] = readers

    status('SECONDARY reader counts and pools')
    arms = {'real': real, 'gaussian': gaussian, 'timing': timing, 'position': position}
    secondary = {'by_release': {name: scored.by_release(v) for name, v in values.items()},
                 'by_block': {name: [float(v[b].mean()) for b in scored.blocks] for name, v in values.items()},
                 'all_sentence_pool': {name: scored.across(scored.average(p)) for name, p in arms.items()},
                 'reader_curve': {name: scored.curve(arms[name], counts, draws, seed) for name in ('real', 'gaussian')}}
    all_shuffled = [scored.across(scored.average(p)) for p in shuffled]
    secondary['all_sentence_pool']['class_shuffled'] = {release: {f'{w}_way': float(np.mean([s[release][f'{w}_way'] for s in all_shuffled])) for w in WAYS}
                                                        for release in layout.releases}
    status('SECONDARY time-binned input')
    secondary['time_binned'] = {}
    for name in ('real', 'gaussian'):
        _, predicted, _ = run_arm(layout, eeg(features[f'binned_{name}']), true, alphas, seed)
        secondary['time_binned'][name] = float(scored.within(scored.average(predicted[0])).mean())
    status('SECONDARY fit on sentence means')
    secondary['fit_on_means'] = {}
    for name in ('real', 'gaussian'):
        _, predicted, _ = run_arm(layout, eeg(features[name]), true, alphas, seed, fit_means=True)
        secondary['fit_on_means'][name] = float(scored.within(scored.average(predicted[0])).mean())
    status('SECONDARY new readers')
    fit_readers, score_readers = half_readers(layout, seed)
    new_rows, predicted, _ = run_arm(layout, eeg(features['real']), lambda i: [text] + shuffled_tables(layout, i, text, shuffles, seed), alphas, seed,
                                     fit_readers=fit_readers, score_readers=score_readers)
    new_scored = Scored(layout, new_rows, text)
    new_real = new_scored.within(new_scored.average(predicted[0]))
    new_shuffled = np.mean([new_scored.within(new_scored.average(p)) for p in predicted[1:]], axis=0)
    _, predicted, _ = run_arm(layout, eeg(features['gaussian']), true, alphas, seed, fit_readers=fit_readers, score_readers=score_readers)
    new_gaussian = new_scored.within(new_scored.average(predicted[0]))
    secondary['new_readers'] = {'real': float(new_real.mean()), 'gaussian': float(new_gaussian.mean()), 'class_shuffled': float(new_shuffled.mean()),
                                'real_minus_gaussian': resample_gap(new_real - new_gaussian, new_scored.block_of, iterations, seed),
                                'real_minus_class_shuffled': resample_gap(new_real - new_shuffled, new_scored.block_of, iterations, seed),
                                'fit_readers': int(len(set(layout.reader[fit_readers]))), 'scored_readers': int(len(set(layout.reader[score_readers])))}
    return {'decision': decision, 'secondary': secondary, 'selection': selection, 'values': values}


__all__ = ['FIRST', 'Fold', 'Layout', 'SECOND', 'Scored', 'Solver', 'THIRD', 'fit_fold', 'fold_audit', 'half_readers', 'read_outcome', 'resample_gap',
           'resample_readers', 'run_arm', 'run_test', 'scale_by_reader', 'shuffled_tables', 'timing_options', 'within_score']
