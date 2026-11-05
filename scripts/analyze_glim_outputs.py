"""Compare GLIM's published test outputs with neighbouring text on the same queries. No EEG is read and nothing is trained.

Run with `python -I` and the CSV folder as an argument; the CSV files are third-party data and are only parsed.
"""
import csv
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
os.environ['HF_HUB_OFFLINE'] = '1'
import numpy as np
from sentence_transformers import SentenceTransformer
from eeg_decoding.data.dataset import read_manifest
from eeg_decoding.evaluation.block_holdout import _macro, bootstrap_gap
from eeg_decoding.evaluation.nway import expected_top1, score_counts
from eeg_decoding.evaluation.position_bridge import bridge_scope, identity
from eeg_decoding.evaluation.proximity import excluded_positions
from eeg_decoding.evaluation.retrieval import normalize
from eeg_decoding.training.alignment_train import TextCache
from eeg_decoding.utils.logging import write_json

PLAN = ROOT/'results/position_bridge_v4_20261105/plan.json'
OUTPUT = ROOT/'results/glim_output_check_20261106'
FILES = {'glim': 'wandb_export_gen_samples_glim.csv', 'noise_input': 'wandb_export_gen_samples_noise_input_test.csv',
         'prompt_free': 'wandb_export_gen_samples_prompt_free_test.csv'}
MODEL, REVISION = 'sentence-transformers/all-MiniLM-L6-v2', '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
WAYS = (2, 4, 10, 24)
TRIALS = 10000
LAGS = ((1, 2), (3, 5), (6, 10), (11, 10**9))
csv.field_size_limit(10**8)


class Row:
    def __init__(self, release, subject, sentence_id, key):
        self.dataset, self.subject, self.sentence_id, self.key = release, subject, sentence_id, key
        self.subject_key = f'{release}:{subject}'


def squash(sentence):
    return re.sub(r'[^a-z0-9]+', '', sentence.lower())


def read_rows(path):
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def trial_scores(predicted, targets, trials=TRIALS, seed=42):
    """SemKey-style trials; the same draws serve every predictor, so the rows stay paired."""
    output = {}
    for way in WAYS:
        rng = np.random.default_rng(seed)
        total = 0.
        for _ in range(trials):
            picks = rng.choice(len(predicted), way, replace=False)
            total += np.mean((predicted[picks] @ targets[picks].T).argmax(1) == np.arange(way))
        output[f'{way}_way'] = total / trials
    return output


def exact_scores(predicted, rows, pool_vectors, pool_keys, way=10):
    """Expected N-way per query against the unique scored sentences; ties are misses."""
    place = {k: i for i, k in enumerate(pool_keys)}
    below, _, _ = score_counts(predicted @ pool_vectors.T, np.array([place[r.key] for r in rows]))
    return expected_top1(below, len(pool_keys), way)


def interval(values, rows):
    report = bootstrap_gap(values, rows, 10000, 42, level=95)
    return {'estimate': _macro(values, rows), 'reader_interval': report['reader_interval'], 'sentence_interval': report['sentence_interval']}


def closest_text(rows, train_ids, lookup, count=2):
    ordered = {d: np.array(sorted(ids)) for d, ids in train_ids.items()}
    output = []
    for r in rows:
        ids = ordered[r.dataset]
        picks = ids[np.lexsort((ids, np.abs(ids - r.sentence_id)))[:count]]
        output.append(np.mean([lookup[r.dataset, int(i)] for i in picks], axis=0))
    return normalize(np.array(output))


def position_readout(vectors, rows, train_ids, lookup, iterations=10000, seed=42):
    """Where in the list does the closest training sentence sit? Sentence means, correlated with true position per release."""
    output = {}
    for release, ids in train_ids.items():
        ordered = sorted(ids)
        bank = normalize(np.array([lookup[release, i] for i in ordered]))
        mine = [i for i, r in enumerate(rows) if r.dataset == release]
        picked = np.array(ordered)[(vectors[mine] @ bank.T).argmax(1)]
        by_sentence = {}
        for i, value in zip(mine, picked):
            by_sentence.setdefault(rows[i].sentence_id, []).append(value)
        truth = np.array(sorted(by_sentence), dtype=float)
        means = np.array([np.mean(by_sentence[int(t)]) for t in truth])
        rng = np.random.default_rng(seed)
        draws = [np.corrcoef(means[p], truth[p])[0, 1] for p in (rng.integers(len(truth), size=len(truth)) for _ in range(iterations))]
        output[release] = {'sentences': len(truth), 'correlation': float(np.corrcoef(means, truth)[0, 1]), 'interval_95': np.nanpercentile(draws, [2.5, 97.5]).tolist(),
                           'mean_error': float(np.mean(np.abs(picked - np.array([rows[i].sentence_id for i in mine]))))}
    return output


def similarity_by_lag(vectors, rows, train_ids, lookup):
    output = {}
    banks = {d: (np.array(sorted(ids)), normalize(np.array([lookup[d, i] for i in sorted(ids)]))) for d, ids in train_ids.items()}
    sums = {lag: [] for lag in LAGS}
    for vector, r in zip(vectors, rows):
        ids, bank = banks[r.dataset]
        values, distance = bank @ vector, np.abs(ids - r.sentence_id)
        for lag in LAGS:
            chosen = (distance >= lag[0]) & (distance <= lag[1])
            if chosen.any():
                sums[lag].append(float(values[chosen].mean()))
    for lag in LAGS:
        output[f'{lag[0]}+' if lag[1] > 1000 else f'{lag[0]}-{lag[1]}'] = float(np.mean(sums[lag]))
    return output


def main(folder):
    if OUTPUT.exists():
        raise FileExistsError('Use a fresh output directory')
    plan = json.loads(PLAN.read_text(encoding='utf-8'))['plan']
    records = [r for path in plan['cohort']['manifests'] for r in read_manifest(ROOT/path)]
    zone = excluded_positions(records, plan['seed'], **plan['protected']['v2_test_zone'])
    _, _, closed, v1_test = bridge_scope(records, plan['seed'], zone)
    protected = closed | v1_test
    text = TextCache(ROOT/plan['text_cache'])
    by_text, key_of, listing = {}, {}, {}
    for r in records:
        by_text.setdefault((r.dataset, squash(r.text)), set()).add(int(r.sentence_id))
        key_of[identity(r)] = r.key
        listing.setdefault(r.dataset, set()).add(int(r.sentence_id))
    lookup = {i: text.get([k])[0] for i, k in key_of.items()}
    outputs = {name: read_rows(Path(folder)/file) for name, file in FILES.items()}
    base = outputs['glim']
    if any([(a['Subject'], a['Input Text']) for a in outputs[n]] != [(a['Subject'], a['Input Text']) for a in base] for n in outputs):
        raise ValueError('The output files do not list the same queries in the same order')
    test_ids = {d: set() for d in listing}
    kept, unmatched = [], set()
    counts = dict(rows=len(base), normal_reading_list_rows=0, unmatched_rows=0, protected_rows=0, ambiguous_rows=0)
    for index, row in enumerate(base):
        if row['Raw Task Key'] != 'task2':
            continue
        counts['normal_reading_list_rows'] += 1
        release = row['Dataset'].lower()
        ids = by_text.get((release, squash(row['Input Text'])))
        if not ids:
            counts['unmatched_rows'] += 1
            unmatched.add(row['Input Text'][:60])
            continue
        if len(ids) > 1:
            counts['ambiguous_rows'] += 1
            continue
        i = next(iter(ids))
        if (release, i) in protected:
            counts['protected_rows'] += 1
            continue
        kept.append((index, Row(release, row['Subject'], i, key_of[release, i])))
    # Any text GLIM held out in another task also stays out of the neighbour bank.
    for row in base:
        release = row['Dataset'].lower()
        test_ids[release] |= by_text.get((release, squash(row['Input Text'])), set())
    indices, rows = [k[0] for k in kept], [k[1] for k in kept]
    train_ids = {d: {i for i in ids if i not in test_ids[d] and (d, i) not in protected} for d, ids in listing.items()}
    model = SentenceTransformer(MODEL, revision=REVISION, device='cpu')

    def encode(sentences):
        return normalize(model.encode(sentences, batch_size=64, show_progress_bar=False, convert_to_numpy=True).astype(np.float64))

    pool_keys = sorted({r.key for r in rows})
    pool = normalize(text.get(pool_keys).astype(np.float64))
    targets = normalize(text.get([r.key for r in rows]).astype(np.float64))
    predictors = {name: encode([table[i]['Generated Text'] for i in indices]) for name, table in outputs.items()}
    predictors['neighbour_text'] = closest_text(rows, train_ids, lookup)
    predictors['target_text'] = targets
    result = {'note': 'Exploratory. GLIM outputs come from its published CSV files; it is not known which checkpoint produced them.', 'counts': counts,
              'scored_observations': len(rows), 'scored_sentences': len(pool_keys), 'readers': len({r.subject_key for r in rows}),
              'sentences_by_release': {d: len({r.sentence_id for r in rows if r.dataset == d}) for d in listing},
              'unmatched_examples': sorted(unmatched), 'scores': {}, 'readout': {}, 'similarity_by_lag': {}}
    exact = {}
    for name, vectors in predictors.items():
        result['scores'][name] = {'trials': trial_scores(vectors, targets)} if name != 'target_text' else {}
        exact[name] = exact_scores(vectors, rows, pool, pool_keys)
        result['scores'][name]['exact_10_way'] = _macro(exact[name], rows)
        result['readout'][name] = position_readout(vectors, rows, train_ids, lookup)
        result['similarity_by_lag'][name] = similarity_by_lag(vectors, rows, train_ids, lookup)
    result['comparisons'] = {'neighbour_minus_glim': interval(exact['neighbour_text'] - exact['glim'], rows),
                             'glim_minus_noise_input': interval(exact['glim'] - exact['noise_input'], rows),
                             'prompt_free_minus_noise_input': interval(exact['prompt_free'] - exact['noise_input'], rows)}
    # GLIM's validation sentences cannot be identified, so drop a matching share of neighbours at random as a sensitivity check.
    drop = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        thinned = {d: {i for i in sorted(ids) if rng.random() > .11} for d, ids in train_ids.items()}
        drop.append(_macro(exact_scores(closest_text(rows, thinned, lookup), rows, pool, pool_keys), rows))
    result['neighbour_without_validation_share'] = {'mean': float(np.mean(drop)), 'min': float(np.min(drop)), 'max': float(np.max(drop))}
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT/'scores.json', result)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
