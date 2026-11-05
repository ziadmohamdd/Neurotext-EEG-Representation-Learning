"""Sanity checks on the v1 neighbouring-text baseline. No EEG, no training, no test sentence scored."""
import json
from pathlib import Path
import re
import numpy as np
from eeg_decoding.data.dataset import read_manifest
from eeg_decoding.evaluation.block_holdout import _macro, bootstrap_gap
from eeg_decoding.evaluation.proximity import closest_text_prediction
from eeg_decoding.evaluation.sentence_holdout import report_scores, score_queries, sentence_holdout_split
from eeg_decoding.training.alignment_train import TextCache
from eeg_decoding.utils.logging import write_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'results/sentence_holdout_pilot_v1_20261105'
OUTPUT = ROOT/'results/v1_neighbour_text_checks_20261105'
SEEDS = 20


def tokens(sentence):
    return set(re.findall(r"[a-z0-9']+", sentence.lower()))


def containment(a, b):
    return len(a & b) / max(1, min(len(a), len(b)))


def main():
    if OUTPUT.exists():
        raise FileExistsError('Use a fresh output directory')
    plan = json.loads((SOURCE/'plan.json').read_text(encoding='utf-8'))['plan']
    records = [r for path in plan['cohort']['manifests'] for r in read_manifest(ROOT/path)]
    splits = sentence_holdout_split(records, plan['seed'])
    text = TextCache(ROOT/plan['text_cache'])
    rows = splits['val']
    train_ids, lookup, words = {}, {}, {}
    for r in splits['train']:
        train_ids.setdefault(r.dataset, set()).add(int(r.sentence_id))
        lookup[r.dataset, int(r.sentence_id)] = text.get([r.key])[0]
        words[r.dataset, int(r.sentence_id)] = tokens(r.text)
    val_ids = {(r.dataset, int(r.sentence_id)) for r in rows}
    train_pairs = {(d, i) for d, ids in train_ids.items() for i in ids}
    train_keys, val_keys = {r.key for r in splits['train']}, {r.key for r in rows}

    def primary(predicted):
        return score_queries(predicted, rows, text, (2, 4, 10), 10, 20)

    real = primary(closest_text_prediction(rows, train_ids, lookup, 2))
    values = real['expected'][10]
    overlaps, cosines = [], []
    for r in rows:
        ids = sorted(train_ids[r.dataset], key=lambda i: (abs(i - int(r.sentence_id)), i))[:2]
        overlaps.append(max(containment(tokens(r.text), words[r.dataset, i]) for i in ids))
        own = text.get([r.key])[0]
        cosines.append(max(float(own @ lookup[r.dataset, i] / np.linalg.norm(own) / np.linalg.norm(lookup[r.dataset, i])) for i in ids))
    overlaps, cosines = np.array(overlaps), np.array(cosines)
    # The null keeps the pool and the two-sentence average but picks training sentences at random.
    null = []
    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        picked = np.array([np.mean([lookup[r.dataset, i] for i in rng.choice(sorted(train_ids[r.dataset]), 2, replace=False)], axis=0) for r in rows])
        null.append(_macro(primary(picked)['expected'][10], rows))
    subsets = {}
    for limit in (.8, .5, .3):
        chosen = overlaps < limit
        subset = [r for r, keep in zip(rows, chosen) if keep]
        subsets[f'word_overlap_below_{limit}'] = {'observations': int(chosen.sum()), 'sentences': len({r.key for r in subset}),
                                                  'readers': len({r.subject_key for r in subset}), '10_way': _macro(values[chosen], subset) if subset else None}
    for limit in (.7, .5):
        chosen = cosines < limit
        subset = [r for r, keep in zip(rows, chosen) if keep]
        subsets[f'cosine_below_{limit}'] = {'observations': int(chosen.sum()), 'sentences': len({r.key for r in subset}),
                                            'readers': len({r.subject_key for r in subset}), '10_way': _macro(values[chosen], subset) if subset else None}
    sentence_scores = {}
    for r, value, overlap, cosine in zip(rows, values, overlaps, cosines):
        sentence_scores.setdefault((r.dataset, int(r.sentence_id)), (float(value), float(overlap), float(cosine), r.text))
    interval = bootstrap_gap(values, rows, 10000, 42, level=95)
    result = {'observations': len(rows), 'sentences': len(val_ids), 'train_val_id_overlap': len(val_ids & train_pairs), 'train_val_key_overlap': len(train_keys & val_keys),
              'closest_2': report_scores(real, rows, 10)['metrics'], 'interval_95': {k: interval[k] for k in ('reader_interval', 'sentence_interval')},
              'random_training_text_10_way': {'mean': float(np.mean(null)), 'min': float(np.min(null)), 'max': float(np.max(null)), 'seeds': SEEDS},
              'neighbour_overlap': {'median_word_overlap': float(np.median(overlaps)), 'median_cosine': float(np.median(cosines)),
                                    'share_overlap_at_least_0.8': float(np.mean(overlaps >= .8))},
              'subsets': subsets}
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT/'checks.json', result)
    print(json.dumps(result, indent=1))
    print('Top sentences by overlap')
    for key, item in sorted(sentence_scores.items(), key=lambda kv: -kv[1][1])[:6]:
        print(key, round(item[0], 3), round(item[1], 2), round(item[2], 2), item[3][:80])
    print('Lowest overlap')
    for key, item in sorted(sentence_scores.items(), key=lambda kv: kv[1][1])[:6]:
        print(key, round(item[0], 3), round(item[1], 2), round(item[2], 2), item[3][:80])


if __name__ == '__main__':
    main()
