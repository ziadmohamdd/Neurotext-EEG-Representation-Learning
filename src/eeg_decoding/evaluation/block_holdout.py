"""Hold out contiguous stimulus-list blocks so reading order cannot stand in for meaning."""
from collections import Counter, defaultdict
import numpy as np
from .sentence_holdout import (NEAR_DUPLICATE_JACCARD, TOKEN, report_scores, reader_macro, score_queries, sentence_groups,
                               word_count)


def _place_blocks(count, block, buffer, blocks, forbidden, rng, attempts=1000000):
    for _ in range(attempts):
        starts = np.sort(rng.integers(0, count - block + 1, size=blocks))
        if np.any(np.diff(starts) < block + buffer):
            continue
        if not any(position in forbidden for s in starts for position in range(s, s + block)):
            return [int(s) for s in starts]
    raise ValueError("No valid block placement")


def block_holdout_split(records, seed=42, block=25, buffer=10, val_blocks=2, test_blocks=2):
    """Return train/val/test rows, each held-out row's block, and an audit.

    Buffer sentences join no split. A sentence that repeats in its list, has a
    near-duplicate, or occurs in both releases never enters a held-out block.
    """
    groups, _ = sentence_groups(records)
    group_of = {k: i for i, g in enumerate(groups) for k in g}
    releases, listing = defaultdict(set), defaultdict(dict)
    for r in records:
        releases[group_of[r.key]].add(r.dataset)
        listing[r.dataset][int(r.sentence_id)] = r.key
    rng = np.random.default_rng(seed)
    label = {}
    total = val_blocks + test_blocks
    for release in sorted(listing):
        order = sorted(listing[release])
        repeats = Counter(group_of[listing[release][i]] for i in order)
        forbidden = {p for p, i in enumerate(order)
                     if len(releases[group_of[listing[release][i]]]) > 1 or len(groups[group_of[listing[release][i]]]) > 1
                     or repeats[group_of[listing[release][i]]] > 1}
        starts = _place_blocks(len(order), block, buffer, total, forbidden, rng)
        names = ["val"] * val_blocks + ["test"] * test_blocks
        assigned = {starts[i]: names[n] for n, i in enumerate(rng.permutation(total))}
        for position, identity in enumerate(order):
            value = "train"
            for start, name in assigned.items():
                if start <= position < start + block:
                    value = f"{name}:{release}:{start}"
                    break
                if start - buffer <= position < start + block + buffer:
                    value = "buffer"
            label[release, identity] = value
    held_groups = {}
    for r in records:
        value = label[r.dataset, int(r.sentence_id)]
        if value.startswith(("val:", "test:")):
            name = value.split(":")[0]
            if held_groups.setdefault(group_of[r.key], name) != name:
                raise ValueError("A sentence group crosses validation and test")
    splits, blocks, dropped = {"train": [], "val": [], "test": []}, {}, 0
    buffered = 0
    for r in records:
        value = label[r.dataset, int(r.sentence_id)]
        if value == "buffer":
            buffered += 1
        elif value == "train":
            if group_of[r.key] in held_groups:
                dropped += 1
            else:
                splits["train"].append(r)
        else:
            splits[value.split(":")[0]].append(r)
            blocks[r.dataset, r.subject, r.session, r.task, r.sentence_id] = value
    check_block_split(splits, group_of)
    return splits, blocks, {"buffer_observations": buffered, "near_duplicate_training_observations_dropped": dropped}


def check_block_split(splits, group_of):
    if any(not rows for rows in splits.values()):
        raise ValueError("Empty split")
    seen = {name: {group_of[r.key] for r in rows} for name, rows in splits.items()}
    if seen["train"] & seen["val"] or seen["train"] & seen["test"] or seen["val"] & seen["test"]:
        raise ValueError("Sentence group leakage between splits")


def leakage_audit(splits, blocks, records, text, buffer, jaccard=NEAR_DUPLICATE_JACCARD, containment=.8, cosine=.9):
    """Check the real split for leaks that scores alone would not reveal.

    Returns every failed check, so a run can refuse to start. Text is compared
    pairwise, because a near-copy in training would silently inflate held-out scores.
    """
    failures, summary = [], {}
    names = ("train", "val", "test")
    keys = {name: {r.key for r in splits[name]} for name in names}
    identities = {name: {(r.dataset, r.subject, r.session, r.task, r.sentence_id) for r in splits[name]} for name in names}
    for i, a in enumerate(names):
        for b in names[i+1:]:
            if keys[a] & keys[b]:
                failures.append(f"Sentence appears in {a} and {b}")
            if identities[a] & identities[b]:
                failures.append(f"Observation appears in {a} and {b}")
            if {r.signal_path for r in splits[a]} & {r.signal_path for r in splits[b]}:
                failures.append(f"Signal file appears in {a} and {b}")
    wording = {}
    for r in records:
        wording[r.key] = set(TOKEN.findall(r.text.lower()))
    held = sorted(keys["val"] | keys["test"])
    # Buffer sentences are compared too, so nothing near-copied can hide outside training.
    others = sorted(set(wording) - set(held))
    worst_jaccard, worst_containment = 0., 0.
    for a in held:
        for b in others + [k for k in held if k != a and (a in keys["val"]) != (k in keys["val"])]:
            shared = len(wording[a] & wording[b])
            worst_jaccard = max(worst_jaccard, shared / max(len(wording[a] | wording[b]), 1))
            worst_containment = max(worst_containment, shared / max(min(len(wording[a]), len(wording[b])), 1))
    summary.update(max_token_jaccard=worst_jaccard, max_token_containment=worst_containment)
    if worst_jaccard >= jaccard:
        failures.append(f"Near-duplicate wording across a split boundary: Jaccard {worst_jaccard:.3f}")
    if worst_containment >= containment:
        failures.append(f"One sentence nearly contains another across a split boundary: {worst_containment:.3f}")
    target = text.get(sorted(keys["train"]))
    target = target / np.linalg.norm(target, axis=1, keepdims=True)
    query = text.get(held)
    query = query / np.linalg.norm(query, axis=1, keepdims=True)
    closest = float((query @ target.T).max())
    summary["max_text_cosine_held_out_to_train"] = closest
    if closest >= cosine:
        failures.append(f"Held-out sentence embedding nearly matches training: cosine {closest:.3f}")
    gaps = []
    for release in sorted({r.dataset for r in records}):
        train_ids = {int(r.sentence_id) for r in splits["train"] if r.dataset == release}
        for name in ("val", "test"):
            for r in splits[name]:
                if r.dataset == release:
                    gaps.append(min((abs(int(r.sentence_id) - i) for i in train_ids), default=10**9))
    summary["min_list_distance_held_out_to_train"] = min(gaps)
    if min(gaps) <= buffer:
        failures.append(f"Training is {min(gaps)} list positions from a held-out sentence; buffer is {buffer}")
    readers = {r.subject_key for r in splits["train"]}
    for name in ("val", "test"):
        if not {r.subject_key for r in splits[name]} <= readers:
            failures.append(f"Reader in {name} has no training statistics")
    if any(not value.startswith(("val:", "test:")) for value in blocks.values()):
        failures.append("A scored block is not validation or test")
    summary["failures"] = failures
    return summary


def row_block(blocks, r):
    return blocks[r.dataset, r.subject, r.session, r.task, r.sentence_id]


def within_block_scores(predicted, rows, blocks, text, way=10):
    """Score each query only against sentences in its own contiguous block."""
    names = [row_block(blocks, r) for r in rows]
    output = np.zeros(len(rows))
    for name in sorted(set(names)):
        indices = np.array([i for i, n in enumerate(names) if n == name])
        part = [rows[i] for i in indices]
        scored = score_queries(predicted[indices], part, text, ways=(way,), primary=way, matched_size=way)
        output[indices] = scored["expected"][way]
    return output


def position_features(rows, terms, positions=None):
    """Release flag plus a Fourier basis of list position; contains no EEG."""
    if positions is None:
        positions = {}
        for r in rows:
            low, high = positions.get(r.dataset, (int(r.sentence_id), int(r.sentence_id)))
            positions[r.dataset] = min(low, int(r.sentence_id)), max(high, int(r.sentence_id))
    output = []
    releases = sorted(positions)
    for r in rows:
        low, high = positions[r.dataset]
        p = (int(r.sentence_id) - low) / max(high - low, 1)
        row = [float(r.dataset == release) for release in releases[1:]]
        for k in range(1, terms + 1):
            row += [np.sin(2 * np.pi * k * p), np.cos(2 * np.pi * k * p)]
        output.append(row)
    return np.array(output, dtype=np.float64)


def list_ranges(records):
    positions = {}
    for r in records:
        low, high = positions.get(r.dataset, (int(r.sentence_id), int(r.sentence_id)))
        positions[r.dataset] = min(low, int(r.sentence_id)), max(high, int(r.sentence_id))
    return positions


def bootstrap_gap(gap, rows, iterations=10000, seed=42, level=98.33):
    """Percentile interval of the reader-macro gap, resampling readers and sentences."""
    gap = np.asarray(gap, dtype=float)
    readers, sentences = sorted({r.subject_key for r in rows}), sorted({r.key for r in rows})
    r_index, s_index = {k: i for i, k in enumerate(readers)}, {k: i for i, k in enumerate(sentences)}
    total, count = np.zeros((len(readers), len(sentences))), np.zeros((len(readers), len(sentences)))
    for value, r in zip(gap, rows):
        total[r_index[r.subject_key], s_index[r.key]] += value
        count[r_index[r.subject_key], s_index[r.key]] += 1
    rng = np.random.default_rng(seed)
    bounds = [(100 - level) / 2, 100 - (100 - level) / 2]
    means = total.sum(1) / count.sum(1)
    by_reader = means[rng.integers(len(readers), size=(iterations, len(readers)))].mean(1)
    by_sentence = np.zeros(iterations)
    for i in range(iterations):
        weight = np.bincount(rng.integers(len(sentences), size=len(sentences)), minlength=len(sentences))
        denominator = count @ weight
        usable = denominator > 0
        by_sentence[i] = ((total @ weight)[usable] / denominator[usable]).mean()
    return {"level": level, "reader_interval": np.percentile(by_reader, bounds).tolist(),
            "sentence_interval": np.percentile(by_sentence, bounds).tolist(), "iterations": iterations}


def _macro(values, rows):
    return float(np.mean(list(reader_macro(values, rows).values())))


def apply_block_gate(real, comparators, within_real, within_gaussian, rows, minimum_gap=.02,
                     readers_positive=20, iterations=10000, level=98.33):
    """Frozen v2 gate; every score is a per-query expected accuracy."""
    real = np.asarray(real, dtype=float)
    gaps = {name: _macro(real - np.asarray(values, dtype=float), rows) for name, values in comparators.items()}
    noise_gap = real - np.asarray(comparators["gaussian"], dtype=float)
    per_reader = reader_macro(noise_gap, rows)
    positive = sum(value > 0 for value in per_reader.values())
    within_gap = _macro(np.asarray(within_real, dtype=float) - np.asarray(within_gaussian, dtype=float), rows)
    interval = bootstrap_gap(noise_gap, rows, iterations, level=level)
    checks = {"primary_gaps_at_least_minimum": all(value >= minimum_gap for value in gaps.values()),
              "within_block_gap_at_least_minimum": within_gap >= minimum_gap,
              "readers_positive_against_gaussian": positive >= readers_positive,
              "reader_lower_bound_above_zero": interval["reader_interval"][0] > 0,
              "sentence_lower_bound_above_zero": interval["sentence_interval"][0] > 0}
    return {"passed": all(checks.values()), "checks": checks, "primary_gaps": gaps, "within_block_gap": within_gap,
            "readers_positive": positive, "readers_total": len(per_reader), "bootstrap": interval}


__all__ = ["apply_block_gate", "block_holdout_split", "bootstrap_gap", "list_ranges", "position_features",
           "report_scores", "row_block", "score_queries", "within_block_scores", "word_count"]
