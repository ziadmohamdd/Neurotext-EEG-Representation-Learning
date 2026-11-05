"""Familiar-reader, new-sentence split and scoring for the frozen pilot plan."""
from collections import defaultdict
import re
import numpy as np
from .nway import expected_top1, score_counts
from .retrieval import normalize

NEAR_DUPLICATE_JACCARD = .7
TOKEN = re.compile(r"[a-z0-9]+")


def word_count(text):
    return len(text.split())


def sentence_groups(records):
    """Union sentence keys whose alphanumeric tokens are identical or nearly so."""
    text = {}
    for r in records:
        text[r.key] = r.text
    keys = sorted(text)
    parent = {k: k for k in keys}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    tokens = {k: set(TOKEN.findall(text[k].lower())) for k in keys}
    merged = []
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            union = tokens[a] | tokens[b]
            if not union or len(tokens[a] & tokens[b]) / len(union) >= NEAR_DUPLICATE_JACCARD:
                parent[find(a)] = find(b)
                merged.append((a, b))
    groups = defaultdict(list)
    for k in keys:
        groups[find(k)].append(k)
    return list(groups.values()), merged


def sentence_holdout_split(records, seed=42, fraction=.15):
    """Hold out sentence groups; every reader may appear in every split.

    Groups in both releases stay in training. The rest are stratified by release
    and word-count tertile, so held-out pools are comparable within each release.
    """
    groups, _ = sentence_groups(records)
    text, releases = {}, defaultdict(set)
    for r in records:
        text[r.key] = r.text
        releases[r.key].add(r.dataset)
    assignment = {}
    single = defaultdict(list)
    for group in groups:
        names = {d for k in group for d in releases[k]}
        if len(names) > 1:
            assignment.update({k: "train" for k in group})
        else:
            single[next(iter(names))].append(group)
    rng = np.random.default_rng(seed)
    for release in sorted(single):
        members = single[release]
        lengths = np.array([word_count(text[g[0]]) for g in members])
        cuts = np.quantile(lengths, [1/3, 2/3])
        strata = np.digitize(lengths, cuts)
        for stratum in range(3):
            order = np.flatnonzero(strata == stratum)
            rng.shuffle(order)
            count = round(fraction * len(order))
            for position, index in enumerate(order):
                name = "test" if position < count else "val" if position < 2 * count else "train"
                assignment.update({k: name for k in members[index]})
    splits = {name: [r for r in records if assignment[r.key] == name] for name in ("train", "val", "test")}
    check_sentence_disjoint(splits, groups)
    return splits


def check_sentence_disjoint(splits, groups=None):
    names = list(splits)
    if any(not rows for rows in splits.values()):
        raise ValueError("Empty split")
    for i, a in enumerate(names):
        for b in names[i+1:]:
            if {r.key for r in splits[a]} & {r.key for r in splits[b]}:
                raise ValueError(f"Sentence leakage between {a} and {b}")
    if groups:
        location = {r.key: name for name, rows in splits.items() for r in rows}
        for group in groups:
            if len({location[k] for k in group if k in location}) > 1:
                raise ValueError("Near-duplicate sentences cross a split boundary")


def _matched_pool(key, pool, counts, size):
    others = sorted((k for k in pool if k != key), key=lambda k: (abs(counts[k] - counts[key]), k))
    return [key] + others[:size - 1]


def score_queries(predicted, rows, text, ways=(2, 4, 10, 24), primary=10, matched_size=20):
    """Score each query against held-out sentences from its own release only."""
    predicted = normalize(predicted)
    if len(predicted) != len(rows):
        raise ValueError("Predictions and rows differ")
    n = len(rows)
    ranks, expected = np.zeros(n), {w: np.zeros(n) for w in ways}
    matched = np.zeros(n)
    for release in sorted({r.dataset for r in rows}):
        indices = np.array([i for i, r in enumerate(rows) if r.dataset == release])
        pool = sorted({rows[i].key for i in indices})
        lookup = {k: i for i, k in enumerate(pool)}
        counts = {rows[i].key: word_count(rows[i].text) for i in indices}
        scores = predicted[indices] @ normalize(text.get(pool)).T
        targets = np.array([lookup[rows[i].key] for i in indices])
        below, _, rank = score_counts(scores, targets)
        ranks[indices] = rank
        for way in ways:
            expected[way][indices] = expected_top1(below, len(pool), way)
        subsets = {k: np.array([lookup[c] for c in _matched_pool(k, pool, counts, matched_size)]) for k in pool}
        for row, index in enumerate(indices):
            members = subsets[rows[index].key]
            # The correct sentence is always first in its matched subset.
            local, _, _ = score_counts(scores[row, members][None], [0])
            matched[index] = expected_top1(local, len(members), primary)[0]
    return {"rank": ranks, "expected": expected, "matched": matched}


def reader_macro(values, rows):
    readers = sorted({r.subject_key for r in rows})
    values = np.asarray(values, dtype=float)
    return {s: float(values[[r.subject_key == s for r in rows]].mean()) for s in readers}


def _macro(values, rows):
    return float(np.mean(list(reader_macro(values, rows).values())))


def report_scores(scored, rows, primary=10):
    rank = scored["rank"]
    metrics = {f"{way}_way": _macro(v, rows) for way, v in scored["expected"].items()}
    metrics["matched_word_count_%d_way" % primary] = _macro(scored["matched"], rows)
    metrics.update({f"recall@{k}": _macro(rank <= k, rows) for k in (1, 5, 10)})
    metrics["mrr"] = _macro(1 / rank, rows)
    return {"metrics": metrics, "n_queries": len(rows),
            "primary_per_reader": reader_macro(scored["expected"][primary], rows),
            "queries": [{"identity": [r.dataset, r.subject, r.session, r.task, r.sentence_id],
                         "rank": float(a), "primary": float(b)}
                        for r, a, b in zip(rows, rank, scored["expected"][primary])]}


def bootstrap_gap(gap, rows, iterations=10000, seed=42):
    """95% percentile intervals of the reader-macro gap, resampling each factor."""
    gap = np.asarray(gap, dtype=float)
    readers = sorted({r.subject_key for r in rows})
    sentences = sorted({r.key for r in rows})
    r_index = {k: i for i, k in enumerate(readers)}
    s_index = {k: i for i, k in enumerate(sentences)}
    total, count = np.zeros((len(readers), len(sentences))), np.zeros((len(readers), len(sentences)))
    for value, r in zip(gap, rows):
        total[r_index[r.subject_key], s_index[r.key]] += value
        count[r_index[r.subject_key], s_index[r.key]] += 1
    rng = np.random.default_rng(seed)
    means = total.sum(1) / count.sum(1)
    by_reader = means[rng.integers(len(readers), size=(iterations, len(readers)))].mean(1)
    by_sentence = np.zeros(iterations)
    for i in range(iterations):
        weight = np.bincount(rng.integers(len(sentences), size=len(sentences)), minlength=len(sentences))
        denominator = count @ weight
        usable = denominator > 0
        by_sentence[i] = ((total @ weight)[usable] / denominator[usable]).mean()
    return {"reader_ci95": np.percentile(by_reader, [2.5, 97.5]).tolist(),
            "sentence_ci95": np.percentile(by_sentence, [2.5, 97.5]).tolist(),
            "iterations": iterations}


def apply_gate(real, comparators, rows, minimum_gap=.02, readers_positive=20, iterations=10000):
    """Frozen validation gate; `real` and comparators are per-query primary scores."""
    real = np.asarray(real, dtype=float)
    gaps = {name: _macro(real - np.asarray(values, dtype=float), rows) for name, values in comparators.items()}
    noise_gap = real - np.asarray(comparators["gaussian"], dtype=float)
    per_reader = reader_macro(noise_gap, rows)
    positive = sum(value > 0 for value in per_reader.values())
    intervals = bootstrap_gap(noise_gap, rows, iterations)
    checks = {"gaps_at_least_minimum": all(value >= minimum_gap for value in gaps.values()),
              "readers_positive_against_gaussian": positive >= readers_positive,
              "reader_ci_above_zero": intervals["reader_ci95"][0] > 0,
              "sentence_ci_above_zero": intervals["sentence_ci95"][0] > 0}
    return {"passed": all(checks.values()), "checks": checks, "gaps": gaps,
            "readers_positive": positive, "readers_total": len(per_reader), "bootstrap": intervals}
