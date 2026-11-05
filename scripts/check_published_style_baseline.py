"""Neighbouring text under a GLIM- and SemKey-style split and metric. No EEG. Local normal-reading sentences only, protected sentences removed."""
import json
from pathlib import Path
import numpy as np
from eeg_decoding.data.dataset import read_manifest
from eeg_decoding.evaluation.position_bridge import bridge_scope, identity
from eeg_decoding.evaluation.proximity import closest_text_prediction, excluded_positions
from eeg_decoding.evaluation.retrieval import normalize
from eeg_decoding.evaluation.sentence_holdout import sentence_holdout_split
from eeg_decoding.training.alignment_train import TextCache
from eeg_decoding.utils.logging import write_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'results/position_bridge_v4_20261105/plan.json'
OUTPUT = ROOT/'results/published_style_baseline_20261105'
WAYS = (2, 4, 10, 24)
SEEDS = 20
TRIALS = 1000


def trial_accuracy(predicted, targets, way, trials, seed):
    """SemKey-style: sample rows from the whole held-out set; each row in the trial must pick its own target. Ties go to the first index."""
    rng = np.random.default_rng(seed)
    total = 0.
    for _ in range(trials):
        picks = rng.choice(len(predicted), way, replace=False)
        total += np.mean((predicted[picks] @ targets[picks].T).argmax(1) == np.arange(way))
    return total / trials


def main():
    if OUTPUT.exists():
        raise FileExistsError('Use a fresh output directory')
    plan = json.loads(PLAN.read_text(encoding='utf-8'))['plan']
    records = [r for path in plan['cohort']['manifests'] for r in read_manifest(ROOT/path)]
    zone = excluded_positions(records, plan['seed'], **plan['protected']['v2_test_zone'])
    _, _, closed, v1_test = bridge_scope(records, plan['seed'], zone)
    allowed = [r for r in records if identity(r) not in closed and identity(r) not in v1_test]
    text = TextCache(ROOT/plan['text_cache'])
    lookup = {identity(r): text.get([r.key])[0] for r in allowed}
    rows_out, random_out = [], []
    for seed in range(SEEDS):
        splits = sentence_holdout_split(allowed, seed, .1)
        rows = splits['val']
        train_ids = {}
        for r in splits['train']:
            train_ids.setdefault(r.dataset, set()).add(int(r.sentence_id))
        predicted = normalize(closest_text_prediction(rows, train_ids, lookup, 2))
        targets = normalize(text.get([r.key for r in rows]))
        rows_out.append([trial_accuracy(predicted, targets, way, TRIALS, seed) for way in WAYS])
        rng = np.random.default_rng(seed)
        ids = {d: sorted(v) for d, v in train_ids.items()}
        guess = normalize(np.array([np.mean([lookup[r.dataset, int(i)] for i in rng.choice(ids[r.dataset], 2, replace=False)], axis=0) for r in rows]))
        random_out.append([trial_accuracy(guess, targets, way, TRIALS, seed) for way in WAYS])
    rows_out, random_out = np.array(rows_out), np.array(random_out)
    result = {'note': 'Not the GLIM or SemKey split. Normal-reading sentences only, 10% held out, no EEG.', 'allowed_observations': len(allowed), 'seeds': SEEDS, 'trials': TRIALS,
              'neighbour_text': {f'{w}_way': {'mean': float(rows_out[:, i].mean()), 'min': float(rows_out[:, i].min()), 'max': float(rows_out[:, i].max())} for i, w in enumerate(WAYS)},
              'random_training_text': {f'{w}_way': {'mean': float(random_out[:, i].mean())} for i, w in enumerate(WAYS)}}
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT/'scores.json', result)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
