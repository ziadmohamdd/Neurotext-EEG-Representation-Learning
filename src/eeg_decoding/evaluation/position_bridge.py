"""Position bridge: map EEG only to list position, then copy the text of training sentences there."""
from collections import defaultdict
import numpy as np
from .block_holdout import _macro, bootstrap_gap
from .sentence_holdout import report_scores, score_queries, sentence_holdout_split
from ..models.classical import ridge_path


def identity(r):
    return r.dataset, int(r.sentence_id)


def bridge_scope(records, seed, zone, fraction=.15):
    """The v1 split with the v2 test zone removed. v1 test rows and zone rows are never returned."""
    splits = sentence_holdout_split(records, seed, fraction)
    closed = {(release, i) for release, ids in zone.items() for i in ids}
    return ([r for r in splits['train'] if identity(r) not in closed], [r for r in splits['val'] if identity(r) not in closed],
            closed, {identity(r) for r in splits['test']})


def scope_audit(train_rows, val_rows, closed, v1_test):
    """Name every way a protected or held-out sentence could reach a fit or a score."""
    train, val = {identity(r) for r in train_rows}, {identity(r) for r in val_rows}
    failures = []
    if train & val:
        failures.append('Training and scored sentences share an ID')
    if {r.key for r in train_rows} & {r.key for r in val_rows}:
        failures.append('Training and scored sentences share a key')
    if (train | val) & closed:
        failures.append('A v2 test-zone sentence reached a fit or score')
    if (train | val) & v1_test:
        failures.append('A v1 test sentence reached a fit or score')
    if {r.subject_key for r in val_rows} - {r.subject_key for r in train_rows}:
        failures.append('A scored reader is missing from training')
    return {'failures': failures, 'train_sentences': len(train), 'scored_sentences': len(val),
            'train_observations': len(train_rows), 'scored_observations': len(val_rows)}


def shuffled_position_map(rows, seed):
    """Permute positions between sentences within each release, so one sentence keeps one target."""
    rng = np.random.default_rng(seed)
    output = {}
    for release in sorted({r.dataset for r in rows}):
        ids = sorted({int(r.sentence_id) for r in rows if r.dataset == release})
        output.update({(release, a): float(b) for a, b in zip(ids, rng.permutation(ids))})
    return output


def fit_positions(features, rows, alphas, seed, position_map=None, inner=.15):
    """Ridge per release. The alpha comes from held-back training sentences, then all training sentences refit."""
    rng = np.random.default_rng(seed)
    models, tables = {}, {}
    releases = np.array([r.dataset for r in rows])
    targets = np.array([position_map[identity(r)] if position_map else float(r.sentence_id) for r in rows])[:, None]
    ids = np.array([int(r.sentence_id) for r in rows])
    for release in sorted(set(releases)):
        mine = releases == release
        names = sorted(set(ids[mine]))
        held = set(rng.choice(names, max(1, int(round(inner * len(names)))), replace=False).tolist())
        inside = mine & np.isin(ids, sorted(held))
        outside = mine & ~inside
        best, chosen, table = np.inf, None, []
        for alpha, weights in ridge_path(features[outside], targets[outside], alphas):
            error = float(np.mean(np.abs(predict_with(weights, features[inside]) - targets[inside][:, 0])))
            table.append({'alpha': alpha, 'inner_mae': error})
            if error < best:
                best, chosen = error, alpha
        _, models[release] = next(iter(ridge_path(features[mine], targets[mine], [chosen])))
        tables[release] = {'chosen': chosen, 'grid': table}
    return models, tables


def predict_with(weights, features):
    return (((features - weights['mean']) / weights['scale']) @ weights['coef'].T + weights['intercept'])[:, 0]


def predict_positions(models, features, rows):
    output = np.zeros(len(rows))
    releases = np.array([r.dataset for r in rows])
    for release, weights in models.items():
        mine = releases == release
        output[mine] = predict_with(weights, features[mine])
    return output


def bridge_prediction(positions, rows, train_ids, text_by_identity, count=2):
    """Average the text of the training sentences closest to each predicted position; earlier IDs win ties."""
    ordered = {release: np.array(sorted(ids)) for release, ids in train_ids.items()}
    output = []
    for position, r in zip(positions, rows):
        ids = ordered[r.dataset]
        picks = ids[np.lexsort((ids, np.abs(ids - position)))[:count]]
        output.append(np.mean([text_by_identity[r.dataset, int(i)] for i in picks], axis=0))
    return np.array(output, dtype=np.float32)


def position_quality(positions, rows):
    """Absolute error and correlation per release, with the error of always guessing the middle."""
    output = {}
    releases = np.array([r.dataset for r in rows])
    truth = np.array([float(r.sentence_id) for r in rows])
    for release in sorted(set(releases)):
        mine = releases == release
        output[release] = {'mae': float(np.mean(np.abs(positions[mine] - truth[mine]))),
                           'mae_guessing_middle': float(np.mean(np.abs(np.median(truth[mine]) - truth[mine]))),
                           'correlation': float(np.corrcoef(positions[mine], truth[mine])[0, 1])}
    return output


def bridge_reading(real, shuffled, reference, rows, iterations=10000, seed=42):
    """Frozen reading: real bridge against shuffled positions and the v1 CNN on the same queries."""
    real, shuffled = np.asarray(real, dtype=float), np.asarray(shuffled, dtype=float)
    score, gap = _macro(real, rows), _macro(real - shuffled, rows)
    interval = bootstrap_gap(real - shuffled, rows, iterations, seed, level=95)
    beats = gap >= .02 and min(interval['reader_interval'][0], interval['sentence_interval'][0]) > 0
    if beats and score >= reference - .02:
        label = 'Position alone could account for the CNN score'
    elif beats:
        label = 'Position explains part of the CNN score'
    else:
        label = 'This route is not supported'
    return {'label': label, 'bridge': score, 'reference': reference, 'gap_over_shuffled': gap, 'beats_shuffled': bool(beats), 'interval': interval}


def missing_runs(records):
    """Runs of sentences absent per reader. Contiguous gaps of block size would support IDs following the session."""
    listing, seen = defaultdict(set), defaultdict(set)
    for r in records:
        listing[r.dataset].add(int(r.sentence_id))
        seen[r.subject_key].add(int(r.sentence_id))
    output = {}
    for reader, ids in sorted(seen.items()):
        runs, start, previous = [], None, None
        for i in sorted(listing[reader.split(':')[0]] - ids):
            if start is None or i != previous + 1:
                if start is not None:
                    runs.append([start, previous])
                start = i
            previous = i
        if start is not None:
            runs.append([start, previous])
        output[reader] = runs
    return output


def random_split_baseline(records, closed, v1_test, text_by_identity, keys_text, fractions, seeds, ways=(2, 4, 10)):
    """Neighbouring text over fresh random splits, using no protected sentence. Contains no EEG."""
    from .proximity import closest_text_prediction
    allowed = [r for r in records if identity(r) not in closed and identity(r) not in v1_test]
    output = {}
    for fraction in fractions:
        values, skipped = [], 0
        for seed in seeds:
            splits = sentence_holdout_split(allowed, seed, fraction)
            rows = splits['val']
            pools = defaultdict(set)
            for r in rows:
                pools[r.dataset].add(r.key)
            if min(len(p) for p in pools.values()) < max(ways):
                skipped += 1
                continue
            train_ids = defaultdict(set)
            for r in splits['train']:
                train_ids[r.dataset].add(int(r.sentence_id))
            predicted = closest_text_prediction(rows, train_ids, text_by_identity, 2)
            scored = score_queries(predicted, rows, keys_text, ways, max(ways), max(ways))
            values.append([report_scores(scored, rows, max(ways))['metrics'][f'{w}_way'] for w in ways])
        output[str(fraction)] = {'skipped': skipped, 'splits': len(values)}
        if values:
            values = np.array(values)
            output[str(fraction)].update({f'{w}_way': {'mean': float(values[:, i].mean()), 'min': float(values[:, i].min()), 'max': float(values[:, i].max())}
                                          for i, w in enumerate(ways)})
    return output


__all__ = ['bridge_prediction', 'bridge_reading', 'bridge_scope', 'fit_positions', 'identity', 'missing_runs', 'position_quality',
           'predict_positions', 'random_split_baseline', 'scope_audit', 'shuffled_position_map']
