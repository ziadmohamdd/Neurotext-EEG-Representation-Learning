"""Compare training on sentences close to or far from the held-out ones in the reading order."""
from collections import Counter, defaultdict
import numpy as np
from .block_holdout import _macro, block_holdout_split, bootstrap_gap
from .sentence_holdout import NEAR_DUPLICATE_JACCARD, TOKEN, sentence_groups


def excluded_positions(records, seed=42, block=25, buffer=10, val_blocks=2, test_blocks=2):
    """List ids of the v2 test blocks and their buffers; these stay out of every fit and score."""
    splits, _, _ = block_holdout_split(records, seed, block, buffer, val_blocks, test_blocks)
    ids = defaultdict(set)
    listing = defaultdict(set)
    for r in records:
        listing[r.dataset].add(int(r.sentence_id))
    for r in splits["test"]:
        ids[r.dataset].add(int(r.sentence_id))
    output = {}
    for release, order in listing.items():
        order = sorted(order)
        place = {identity: p for p, identity in enumerate(order)}
        zone = {p + d for i in ids[release] for p in [place[i]] for d in range(-buffer, buffer + 1)}
        output[release] = {order[p] for p in zone if 0 <= p < len(order)}
    return output


def forbidden_positions(records, containment=.8):
    """List ids that repeat, occur in both releases, have a near-duplicate or nearly contain another sentence."""
    groups, _ = sentence_groups(records)
    group_of = {k: i for i, g in enumerate(groups) for k in g}
    releases, listing = defaultdict(set), defaultdict(dict)
    wording = {}
    for r in records:
        releases[group_of[r.key]].add(r.dataset)
        listing[r.dataset][int(r.sentence_id)] = r.key
        wording[r.key] = set(TOKEN.findall(r.text.lower()))
    keys = sorted(wording)
    contained = set()
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            if len(wording[a] & wording[b]) / max(min(len(wording[a]), len(wording[b])), 1) >= containment:
                contained.update((a, b))
    output = {}
    for release, keys in listing.items():
        repeats = Counter(group_of[k] for k in keys.values())
        output[release] = {i for i, k in keys.items() if len(releases[group_of[k]]) > 1 or len(groups[group_of[k]]) > 1
                           or repeats[group_of[k]] > 1 or k in contained}
    return output


def _half(ids, rng):
    ids = sorted(ids)
    return {int(i) for i in rng.permutation(ids)[:len(ids) // 2]}


def proximity_folds(records, excluded, cycle=42, cluster=7, offsets=(0, 14, 28), near=5, gap=5, seed=42, select_every=3):
    """Return one fold per offset, each holding sentence ids per role and release.

    Near and far conditions share a sentence count in each release. Selection
    clusters are held out exactly like scored clusters, so both shape the lags.
    """
    forbidden = forbidden_positions(records)
    listing = defaultdict(set)
    for r in records:
        listing[r.dataset].add(int(r.sentence_id))
    folds = []
    for fold, offset in enumerate(offsets):
        roles = {name: {} for name in ("select", "score", "near", "far", "near_half", "far_half")}
        for release_index, release in enumerate(sorted(listing)):
            order = sorted(listing[release])
            blocked = {p for p, i in enumerate(order) if i in excluded.get(release, set()) or i in forbidden[release]}
            zone = {p for p, i in enumerate(order) if i in excluded.get(release, set())}
            retained = []
            for start in range(offset, len(order) - cluster + 1, cycle):
                span = range(start, start + cluster)
                members = [p for p in span if p not in blocked]
                if members and not zone & set(span):
                    retained.append(members)
            select = [p for i, m in enumerate(retained) for p in m if i % select_every == select_every - 1]
            score = [p for i, m in enumerate(retained) for p in m if i % select_every != select_every - 1]
            held = select + score
            if not select or not score:
                raise ValueError("A fold has no selection or scored cluster")
            taken = set(held)
            close, distant = [], []
            for p in range(len(order)):
                if p in taken or p in blocked:
                    continue
                lag = min(abs(p - h) for h in held)
                if lag <= near:
                    close.append(order[p])
                elif lag > near + gap:
                    distant.append(order[p])
            if len(close) > len(distant):
                raise ValueError("Far sentences cannot match the near sentence count")
            rng = np.random.default_rng([seed, fold, release_index])
            base = sorted(int(i) for i in rng.permutation(distant)[:len(distant) - len(close)])
            half_near = _half(close, rng)
            half_base = _half(base, rng)
            half_far = _half(distant, rng)
            # Rounding both near parts down can leave one sentence short of the far half.
            spare = sorted(set(base) - half_base)
            while len(half_near) + len(half_base) < len(half_far):
                half_base.add(spare.pop())
            roles["select"][release] = {order[p] for p in select}
            roles["score"][release] = {order[p] for p in score}
            roles["near"][release] = set(close) | set(base)
            roles["far"][release] = set(distant)
            roles["near_half"][release] = half_near | half_base
            roles["far_half"][release] = half_far
            roles.setdefault("near_only", {})[release] = set(close)
        folds.append(roles)
    return folds


def rows_for(records, ids):
    return [r for r in records if int(r.sentence_id) in ids.get(r.dataset, ())]


def proximity_audit(records, folds, excluded, text, near=5, gap=5, ways=10, select_way=4, jaccard=NEAR_DUPLICATE_JACCARD,
                    containment=.8, cosine=.9):
    """Check every fold for leaks that scores alone would not reveal; returns all failures."""
    failures, sizes = [], []
    worst = {"jaccard": 0., "containment": 0., "cosine": -1.}
    wording, ordered = {}, defaultdict(set)
    for r in records:
        wording[r.key] = set(TOKEN.findall(r.text.lower()))
        ordered[r.dataset].add(int(r.sentence_id))
    place = {release: {i: p for p, i in enumerate(sorted(ids))} for release, ids in ordered.items()}
    train_roles = ("near", "far", "near_half", "far_half")
    for fold, roles in enumerate(folds):
        size = {}
        held_rows = rows_for(records, roles["select"]) + rows_for(records, roles["score"])
        held_keys = {r.key for r in held_rows}
        held_paths = {r.signal_path for r in held_rows}
        for release in sorted(ordered):
            held = roles["select"][release] | roles["score"][release]
            if roles["select"][release] & roles["score"][release]:
                failures.append(f"Fold {fold}: a selection sentence is scored in {release}")
            for name in roles:
                if roles[name][release] & excluded.get(release, set()):
                    failures.append(f"Fold {fold}: {name} touches the v2 test zone in {release}")
            for name in train_roles:
                if roles[name][release] & held:
                    failures.append(f"Fold {fold}: {name} contains a held-out sentence in {release}")
            if len(roles["near"][release]) != len(roles["far"][release]):
                failures.append(f"Fold {fold}: near and far differ in size in {release}")
            if len(roles["near_half"][release]) != len(roles["far_half"][release]):
                failures.append(f"Fold {fold}: half conditions differ in size in {release}")
            positions = place[release]
            far_lag = min(abs(positions[i] - positions[h]) for i in roles["far"][release] for h in held)
            if far_lag <= near + gap:
                failures.append(f"Fold {fold}: far sentence {far_lag} positions from held-out in {release}")
            if not roles["near_only"][release] <= roles["near"][release]:
                failures.append(f"Fold {fold}: near condition lost a near sentence in {release}")
            if len(roles["score"][release]) < ways:
                failures.append(f"Fold {fold}: scored pool below {ways} in {release}")
            if len(roles["select"][release]) < select_way:
                failures.append(f"Fold {fold}: selection pool below {select_way} in {release}")
            size[release] = {name: len(roles[name][release]) for name in roles}
        for name in train_roles:
            rows = rows_for(records, roles[name])
            size[f"{name}_observations"] = len(rows)
            keys = {r.key for r in rows}
            if keys & held_keys:
                failures.append(f"Fold {fold}: {name} shares a sentence key with held-out rows")
            if {r.signal_path for r in rows} & held_paths:
                failures.append(f"Fold {fold}: {name} shares a signal file with held-out rows")
            if not {r.subject_key for r in held_rows} <= {r.subject_key for r in rows}:
                failures.append(f"Fold {fold}: a held-out reader has no {name} training statistics")
            for a in held_keys:
                for b in keys:
                    shared = len(wording[a] & wording[b])
                    worst["jaccard"] = max(worst["jaccard"], shared / max(len(wording[a] | wording[b]), 1))
                    worst["containment"] = max(worst["containment"], shared / max(min(len(wording[a]), len(wording[b])), 1))
            query, target = text.get(sorted(held_keys)), text.get(sorted(keys))
            query = query / np.linalg.norm(query, axis=1, keepdims=True)
            target = target / np.linalg.norm(target, axis=1, keepdims=True)
            worst["cosine"] = max(worst["cosine"], float((query @ target.T).max()))
        sizes.append(size)
    if worst["jaccard"] >= jaccard:
        failures.append(f"Near-duplicate wording across a boundary: Jaccard {worst['jaccard']:.3f}")
    if worst["containment"] >= containment:
        failures.append(f"One sentence nearly contains another across a boundary: {worst['containment']:.3f}")
    if worst["cosine"] >= cosine:
        failures.append(f"Held-out embedding nearly matches training: cosine {worst['cosine']:.3f}")
    return {"failures": failures, "worst": worst, "sizes": sizes}


def closest_text_prediction(rows, train_ids, text_by_release, count=2):
    """Predict each query from the text of its closest training sentences; contains no EEG."""
    output = []
    for r in rows:
        identity = int(r.sentence_id)
        ids = sorted(train_ids[r.dataset], key=lambda i: (abs(i - identity), i))[:count]
        output.append(np.mean([text_by_release[r.dataset, i] for i in ids], axis=0))
    return np.array(output, dtype=np.float32)


def proximity_decision(near, far, far_noise, rows, iterations=10000, seed=42):
    """Frozen reading of near minus far; every value is a per-query expected accuracy."""
    near, far, far_noise = (np.asarray(v, dtype=float) for v in (near, far, far_noise))
    gain = _macro(near - far, rows)
    near_score, far_score = _macro(near, rows), _macro(far, rows)
    # A two-sided 95% percentile interval gives the planned one-sided 97.5% lower bound.
    interval = bootstrap_gap(near - far, rows, iterations, seed, level=95)
    lower = interval["reader_interval"][0], interval["sentence_interval"][0]
    clear = min(lower) > 0
    noise_gap = abs(_macro(far - far_noise, rows))
    if gain >= .05 and clear and noise_gap <= .02:
        label = "Closeness explains most of the score"
    elif .02 <= gain < .05 and clear:
        label = "Closeness explains part of the score"
    elif (gain < .02 or not clear) and near_score >= .12:
        label = "Closeness is not the main cause"
    elif (gain < .02 or not clear) and near_score < .12:
        label = "Inconclusive"
    else:
        label = "Inconclusive"
    return {"label": label, "near": near_score, "far": far_score, "gain": gain, "far_minus_far_noise": _macro(far - far_noise, rows),
            "near_share_above_chance": gain / (near_score - .1) if near_score > .1 else None,
            "lower_bounds": {"reader": lower[0], "sentence": lower[1]}, "bootstrap": interval}


__all__ = ["closest_text_prediction", "excluded_positions", "forbidden_positions", "proximity_audit", "proximity_decision",
           "proximity_folds", "rows_for"]
