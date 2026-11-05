"""Bin saved v1 validation scores by list distance to the closest training sentence. Nothing is trained."""
import json
from pathlib import Path
import numpy as np
from eeg_decoding.data.dataset import read_manifest
from eeg_decoding.evaluation.sentence_holdout import sentence_holdout_split
from eeg_decoding.utils.logging import write_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'results/sentence_holdout_pilot_v1_20261105'
OUTPUT = ROOT/'results/v1_proximity_bins_20261105'
BINS = ((1, 1), (2, 2), (3, 3), (4, 10**9))


def closest_lags(train_ids, queries):
    return [min(abs(int(q['identity'][4]) - i) for i in train_ids[q['identity'][0]]) for q in queries]


def bin_scores(queries, lags):
    """Mean within reader, then across the readers present in each bin."""
    output = []
    for low, high in BINS:
        chosen = [q for q, lag in zip(queries, lags) if low <= lag <= high]
        readers = {}
        for q in chosen:
            readers.setdefault((q['identity'][0], q['identity'][1]), []).append(q['primary'])
        sentences = {(q['identity'][0], q['identity'][4]) for q in chosen}
        output.append({'lag_from': low, 'lag_to': None if high == 10**9 else high, 'observations': len(chosen), 'sentences': len(sentences),
                       'readers': len(readers), 'mean_primary': float(np.mean([np.mean(v) for v in readers.values()])) if readers else None})
    return output


def main():
    if OUTPUT.exists():
        raise FileExistsError('Use a fresh output directory')
    plan = json.loads((SOURCE/'plan.json').read_text(encoding='utf-8'))['plan']
    records = [r for path in plan['cohort']['manifests'] for r in read_manifest(ROOT/path)]
    splits = sentence_holdout_split(records, plan['seed'])
    train_ids = {}
    for r in splits['train']:
        train_ids.setdefault(r.dataset, set()).add(int(r.sentence_id))
    result = {}
    for arm in ('cnn_real', 'cnn_gaussian', 'cnn_shuffled', 'ridge_real', 'ridge_gaussian', 'ridge_shuffled', 'timing'):
        queries = json.loads((SOURCE/arm/'validation.json').read_text(encoding='utf-8'))['queries']
        result[arm] = bin_scores(queries, closest_lags(train_ids, queries))
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT/'bins.json', result)
    for arm, rows in result.items():
        print(arm, [(row['lag_from'], row['sentences'], None if row['mean_primary'] is None else round(100 * row['mean_primary'], 2)) for row in rows])


if __name__ == '__main__':
    main()
