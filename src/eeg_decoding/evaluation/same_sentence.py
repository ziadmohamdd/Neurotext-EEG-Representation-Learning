"""Ask whether the same sentence looks alike in EEG across two separately recorded releases. Nothing is fitted to meaning."""
import re
import numpy as np
from .averaged_eeg import scale_by_reader
from .nway import expected_top1, score_counts
from .sentence_holdout import word_count

PRIMARY = 10
WAYS = (2, 4, 10, 24, 48)
FIRST = 'The same sentence matches across releases beyond the controls'
SECOND = 'The match does not exceed word count, reading time or position'
THIRD = 'No same-sentence match found with these features'


def normal_text(text):
    return re.sub(r'\W+', ' ', text).lower().strip()


def shared_pairs(records, closed):
    """Earliest list ID of each text in both releases. A text with any protected copy is left out."""
    copies = {}
    for r in records:
        copies.setdefault(normal_text(r.text), {}).setdefault(r.dataset, set()).add(int(r.sentence_id))
    releases = sorted({r.dataset for r in records})
    pairs = []
    for places in copies.values():
        if len(places) == 2 and not any((release, i) in closed for release, ids in places.items() for i in ids):
            pairs.append(tuple(min(places[release]) for release in releases))
    return releases, sorted(pairs)


class Pairing:
    """Rows of the shared sentences. `rows` holds no protected observation."""
    def __init__(self, rows, pairs, releases, place):
        self.rows, self.pairs, self.releases, self.place = rows, pairs, releases, place
        where = {(release, ids[side]): (number, side) for number, ids in enumerate(pairs) for side, release in enumerate(releases)}
        found = [where.get((r.dataset, int(r.sentence_id)), (-1, -1)) for r in rows]
        self.pair = np.array([f[0] for f in found])
        self.side = np.array([f[1] for f in found])
        names = sorted({r.subject_key for r in rows})
        readers = {name: i for i, name in enumerate(names)}
        self.reader = np.array([readers[r.subject_key] for r in rows])
        self.reader_side = np.array([releases.index(next(r.dataset for r in rows if r.subject_key == name)) for name in names])
        self.count = len(pairs)
        self.release = np.array([releases.index(r.dataset) for r in rows])
        self.identity = np.array([int(r.sentence_id) for r in rows])
        self.words = np.array([word_count(r.text) for r in rows], dtype=float)
        self.outside = self.pair < 0

    def scaled(self, x):
        """Z-score each reader from rows outside the shared sentences."""
        return scale_by_reader(x, self.reader, self.outside)

    def means(self, x, weights=None):
        """One vector per shared sentence and release. A resample that misses a sentence keeps all its readers."""
        weights = np.ones(len(self.rows)) if weights is None else weights
        output = []
        for side in (0, 1):
            total = np.zeros((self.count, x.shape[1]))
            count = np.zeros(self.count)
            chosen = np.flatnonzero((self.side == side) & (weights > 0))
            np.add.at(total, self.pair[chosen], x[chosen] * weights[chosen, None])
            np.add.at(count, self.pair[chosen], weights[chosen])
            for number in np.flatnonzero(count <= 0):
                members = np.flatnonzero((self.side == side) & (self.pair == number))
                total[number], count[number] = x[members].sum(0), len(members)
            output.append(total / count[:, None])
        return output

    def covariates(self, durations):
        """List position, word count and mean duration per sentence on each side; no EEG amplitudes or meaning."""
        output = []
        for side, release in enumerate(self.releases):
            table = np.zeros((self.count, 3))
            for number, ids in enumerate(self.pairs):
                members = (self.side == side) & (self.pair == number)
                table[number] = self.place[release][ids[side]], self.words[members][0], np.mean(np.asarray(durations)[members])
            output.append(table)
        return output


def standardize(x):
    return (x - x.mean(0)) / x.std(0).clip(1e-6)


def remove(x, covariates):
    """Remove what position, its square, word count and duration explain linearly, across the shared sentences."""
    c = standardize(covariates)
    design = np.c_[np.ones(len(c)), c, c[:, 0] ** 2]
    coefficient, *_ = np.linalg.lstsq(design, x, rcond=None)
    return x - design @ coefficient


def similarity(a, b):
    """Cosine with zero vectors left at zero, so they tie and count as misses."""
    unit = lambda m: m / np.where(np.linalg.norm(m, axis=1, keepdims=True) > 1e-10, np.linalg.norm(m, axis=1, keepdims=True), np.inf)
    return unit(np.asarray(a, dtype=np.float64)) @ unit(np.asarray(b, dtype=np.float64)).T


def match(a, b, way=PRIMARY, order=None):
    """Expected N-way per sentence, mean of both directions. `order` re-pairs the second side for a shuffled check."""
    scores = similarity(a, b)
    count = len(a)
    targets = np.arange(count) if order is None else np.asarray(order)
    back = np.argsort(targets)
    forward = expected_top1(score_counts(scores, targets)[0], count, way)
    backward = expected_top1(score_counts(scores.T, back)[0], count, way)
    return (forward + backward[targets]) / 2


def shuffled(a, b, shuffles, seed, way=PRIMARY):
    rng = np.random.default_rng(seed)
    return np.mean([match(a, b, way, rng.permutation(len(a))) for _ in range(shuffles)], axis=0)


def pair_effect(a, b):
    """Mean cosine of a sentence with itself across releases, minus with other sentences."""
    scores = similarity(a, b)
    same = np.diag(scores).mean()
    return float(same), float((scores.sum() - np.trace(scores)) / (scores.size - len(scores)))


def block_effect(x, position, near, shuffles, seed):
    """Within one release, how much more alike sentences in the same stretch of the list are than distant ones."""
    scores = similarity(*(2 * [standardize(x)]))
    upper = np.triu_indices(len(x), 1)
    value = scores[upper]

    def gap(places):
        distance = np.abs(places[upper[0]] - places[upper[1]])
        return value[distance <= near].mean() - value[distance > near].mean()

    rng = np.random.default_rng(seed)
    observed = gap(position)
    null = np.array([gap(rng.permutation(position)) for _ in range(shuffles)])
    return {'near_minus_far': float(observed), 'shuffled_list_mean': float(null.mean()), 'shuffled_list_95th': float(np.percentile(null, 95)),
            'share_of_shuffles_at_least_this_large': float((null >= observed).mean())}


def sentence_table(pairing, x, durations, side):
    """Every usable sentence in one release: averaged EEG, list position and the two length measures."""
    rows = np.flatnonzero(pairing.release == side)
    ids, inverse = np.unique(pairing.identity[rows], return_inverse=True)
    counts = np.bincount(inverse)
    total = np.zeros((len(ids), x.shape[1]))
    np.add.at(total, inverse, x[rows])
    place = pairing.place[pairing.releases[side]]
    length = np.bincount(inverse, np.asarray(durations)[rows]) / counts
    words = np.bincount(inverse, pairing.words[rows]) / counts
    return total / counts[:, None], np.array([place[i] for i in ids], dtype=float), np.c_[length, words]


def interval(values):
    return np.percentile(values, [2.5, 97.5]).tolist()


def resample_sentences(gap, iterations, seed):
    gap = np.asarray(gap, dtype=float)
    rng = np.random.default_rng(seed)
    return {'estimate': float(gap.mean()), 'sentence_interval': interval(gap[rng.integers(len(gap), size=(iterations, len(gap)))].mean(1))}


def reader_weights(reader_side, rng):
    """Resample readers with replacement, separately within each release."""
    weight = np.zeros(len(reader_side))
    for side in (0, 1):
        group = np.flatnonzero(reader_side == side)
        weight[group] = np.bincount(rng.integers(len(group), size=len(group)), minlength=len(group))
    return weight


def read_outcome(gaps, lower_bounds, minimum=.02):
    clear = lambda names: all(gaps[n] >= minimum and all(v > 0 for v in lower_bounds[n].values()) for n in names)
    beats = clear(('gaussian', 'class_shuffled'))
    if not beats:
        label = THIRD
    elif clear(('timing', 'position', 'partial')):
        label = FIRST
    else:
        label = SECOND
    return {'label': label, 'beats_nulls': beats, 'gaps': gaps, 'lower_bounds': lower_bounds, 'minimum_gap': minimum}


def run_test(pairing, features, durations, plan, status=print, iterations=10000, reader_iterations=2000):
    """Every arm and reading in the frozen plan. `features` maps arm names to row-by-feature matrices."""
    seed, shuffles = plan['seed'], plan['evaluation']['shuffles']
    scaled = {name: pairing.scaled(x) for name, x in features.items()}
    cov = pairing.covariates(durations)

    def sides(x, weights=None):
        return [standardize(m) for m in pairing.means(x, weights)]

    def partial_sides(x, weights=None):
        return [standardize(remove(m, c)) for m, c in zip(pairing.means(x, weights), cov)]

    real, noise = sides(scaled['real']), sides(scaled['gaussian'])
    real_partial, noise_partial = partial_sides(scaled['real']), partial_sides(scaled['gaussian'])
    timing = [standardize(c[:, 1:]) for c in cov]
    position = [standardize(np.c_[c[:, 0], c[:, 0] ** 2]) for c in cov]
    values = {'real': match(*real), 'gaussian': match(*noise), 'timing': match(*timing), 'position': match(*position),
              'class_shuffled': shuffled(*real, shuffles, seed), 'real_partial': match(*real_partial), 'gaussian_partial': match(*noise_partial)}
    primary = {name: float(v.mean()) for name, v in values.items()}
    status(f'Primary 10-way across releases: {primary}')
    gaps = {name: primary['real'] - primary[name] for name in ('gaussian', 'class_shuffled', 'timing', 'position')}
    gaps['partial'] = primary['real_partial'] - primary['gaussian_partial']
    intervals = {name: resample_sentences(values['real'] - values[name], iterations, seed) for name in ('gaussian', 'class_shuffled', 'timing', 'position')}
    intervals['partial'] = resample_sentences(values['real_partial'] - values['gaussian_partial'], iterations, seed)

    status('RESAMPLE readers')
    rng = np.random.default_rng(seed)
    reader_gaps = {name: np.zeros(reader_iterations) for name in ('gaussian', 'class_shuffled', 'partial')}
    shuffled_mean = primary['class_shuffled']
    for i in range(reader_iterations):
        row = reader_weights(pairing.reader_side, rng)[pairing.reader]
        value = match(*sides(scaled['real'], row)).mean()
        reader_gaps['gaussian'][i] = value - match(*sides(scaled['gaussian'], row)).mean()
        reader_gaps['class_shuffled'][i] = value - shuffled_mean
        reader_gaps['partial'][i] = match(*partial_sides(scaled['real'], row)).mean() - match(*partial_sides(scaled['gaussian'], row)).mean()
    readers = {name: interval(v) for name, v in reader_gaps.items()}
    lower = {name: {'sentence': intervals[name]['sentence_interval'][0]} for name in gaps}
    for name in readers:
        lower[name]['reader'] = readers[name][0]
    decision = read_outcome(gaps, lower, plan['decision_rule']['minimum_gap_points'] / 100)
    decision.update(primary=primary, intervals=intervals, reader_intervals=readers, sentences=pairing.count)

    status('SECONDARY')
    secondary = {'other_ways': {name: {f'{w}_way': float(match(*pair, way=w).mean()) for w in WAYS if w <= pairing.count}
                                for name, pair in (('real', real), ('gaussian', noise), ('real_partial', real_partial), ('gaussian_partial', noise_partial))},
                 'pair_effect': {'real': pair_effect(*real), 'gaussian': pair_effect(*noise), 'real_partial': pair_effect(*real_partial), 'gaussian_partial': pair_effect(*noise_partial)}}
    secondary['shared_across_releases'] = {name: float(np.corrcoef(cov[0][:, column], cov[1][:, column])[0, 1])
                                           for column, name in enumerate(('list_position', 'word_count', 'mean_duration'))}
    secondary['one_direction'] = {'first_to_second': float(expected_top1(score_counts(similarity(*real), np.arange(pairing.count))[0], pairing.count, PRIMARY).mean()),
                                  'second_to_first': float(expected_top1(score_counts(similarity(*real).T, np.arange(pairing.count))[0], pairing.count, PRIMARY).mean())}
    if 'binned_real' in scaled:
        pair = [sides(scaled['binned_real']), sides(scaled['binned_gaussian'])]
        parts = [partial_sides(scaled['binned_real']), partial_sides(scaled['binned_gaussian'])]
        secondary['time_binned'] = {'real': float(match(*pair[0]).mean()), 'gaussian': float(match(*pair[1]).mean()),
                                    'real_partial': float(match(*parts[0]).mean()), 'gaussian_partial': float(match(*parts[1]).mean())}
    blocks = {}
    near, block_shuffles = plan['evaluation']['block_near_positions'], plan['evaluation']['block_shuffles']
    for side, release in enumerate(pairing.releases):
        blocks[release] = {}
        for name in ('real', 'gaussian'):
            table, place, length = sentence_table(pairing, scaled[name], durations, side)
            order = np.argsort(place)
            table, place, length = table[order], place[order], length[order]
            design = np.c_[np.ones(len(length)), standardize(length)]
            without = table - design @ np.linalg.lstsq(design, table, rcond=None)[0]
            blocks[release][name] = block_effect(table, place, near, block_shuffles, seed)
            blocks[release][f'{name}_length_removed'] = block_effect(without, place, near, block_shuffles, seed)
        blocks[release]['sentences'] = len(place)
    secondary['block_effect'] = blocks
    return {'decision': decision, 'secondary': secondary, 'values': values}


__all__ = ['FIRST', 'Pairing', 'SECOND', 'THIRD', 'block_effect', 'match', 'normal_text', 'pair_effect', 'read_outcome', 'remove', 'run_test', 'shared_pairs', 'shuffled', 'similarity', 'standardize']
