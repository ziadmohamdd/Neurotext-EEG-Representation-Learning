"""Split GLIM's 2-way and N-way scores by whether distractors share the query's task. No EEG; nothing is trained.

Run with `python -I` and the CSV folder as an argument.
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
from eeg_decoding.evaluation.position_bridge import bridge_scope
from eeg_decoding.evaluation.proximity import excluded_positions
from eeg_decoding.evaluation.retrieval import normalize
from eeg_decoding.utils.logging import write_json

PLAN = ROOT/'results/position_bridge_v4_20261105/plan.json'
OUTPUT = ROOT/'results/glim_pair_check_20261106'
FILES = {'glim': 'wandb_export_gen_samples_glim.csv', 'noise_input': 'wandb_export_gen_samples_noise_input_test.csv',
         'prompt_free': 'wandb_export_gen_samples_prompt_free_test.csv'}
MODEL, REVISION = 'sentence-transformers/all-MiniLM-L6-v2', '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
WAYS = (2, 4, 10, 24)
TRIALS = 10000
ITERATIONS = 2000
csv.field_size_limit(10**8)


def squash(sentence):
    return re.sub(r'[^a-z0-9]+', '', sentence.lower())


def read_rows(path):
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def pair_wins(predicted, own, bank):
    """Row by sentence: does the row prefer its own sentence over each other sentence? Ties are misses."""
    scores = predicted @ bank.T
    return scores[np.arange(len(own)), own][:, None] > scores


def sentence_matrix(wins, own, count):
    total, seen = np.zeros((count, count)), np.zeros(count)
    for row, sentence in zip(wins, own):
        total[sentence] += row
        seen[sentence] += 1
    return total / seen[:, None]


def split_score(matrix, same, weights=None):
    """Mean over ordered sentence pairs, weighted by a bootstrap draw when given; the diagonal is never counted."""
    weights = np.ones(len(matrix)) if weights is None else weights
    pair = np.outer(weights, weights)
    np.fill_diagonal(pair, 0)
    same_pair, cross_pair = pair * same, pair * ~same
    return float((matrix * same_pair).sum() / same_pair.sum()), float((matrix * cross_pair).sum() / cross_pair.sum()), float((matrix * pair).sum() / pair.sum())


def distinct_trials(predicted, targets, sentence_rows, candidates, way, seed=42):
    """N-way with one row per sentence, and distractors drawn only from the sentences in `candidates`."""
    rng = np.random.default_rng(seed)
    total = 0.
    for _ in range(TRIALS):
        chosen = rng.choice(candidates, way, replace=False)
        rows = np.array([rng.choice(sentence_rows[s]) for s in chosen])
        total += np.mean((predicted[rows] @ targets[rows].T).argmax(1) == np.arange(way))
    return total / TRIALS


def main(folder):
    if OUTPUT.exists():
        raise FileExistsError('Use a fresh output directory')
    plan = json.loads(PLAN.read_text(encoding='utf-8'))['plan']
    records = [r for path in plan['cohort']['manifests'] for r in read_manifest(ROOT/path)]
    zone = excluded_positions(records, plan['seed'], **plan['protected']['v2_test_zone'])
    _, _, closed, v1_test = bridge_scope(records, plan['seed'], zone)
    protected_text = {squash(r.text) for r in records if (r.dataset, int(r.sentence_id)) in closed | v1_test}
    outputs = {name: read_rows(Path(folder)/file) for name, file in FILES.items()}
    base = outputs['glim']
    keep = [i for i, row in enumerate(base) if squash(row['Input Text']) not in protected_text]
    model = SentenceTransformer(MODEL, revision=REVISION, device='cpu')

    def encode(sentences):
        return normalize(model.encode(sentences, batch_size=64, show_progress_bar=False, convert_to_numpy=True).astype(np.float64))

    sentences = sorted({base[i]['Input Text'] for i in keep})
    place = {s: i for i, s in enumerate(sentences)}
    bank = encode(sentences)
    own = np.array([place[base[i]['Input Text']] for i in keep])
    group_of = {}
    for i in keep:
        group_of[place[base[i]['Input Text']]] = f"{base[i]['Dataset']}_{base[i]['Raw Task Key']}"
    groups = np.array([group_of[i] for i in range(len(sentences))])
    same = groups[:, None] == groups[None, :]
    vectors = {name: encode([table[i]['Generated Text'] for i in keep]) for name, table in outputs.items()}
    # A no-EEG predictor that knows only the query's task: the mean text of the other test sentences in that task.
    label_only = np.zeros((len(keep), bank.shape[1]))
    for row, sentence in enumerate(own):
        members = (groups == groups[sentence]) & (np.arange(len(sentences)) != sentence)
        label_only[row] = bank[members].mean(0)
    vectors['task_label_only'] = normalize(label_only)
    sentence_rows = {s: np.flatnonzero(own == s) for s in range(len(sentences))}
    targets = bank[own]
    result = {'note': 'Exploratory. Rows are weighted equally, as in SemKey trials. One row per sentence in each N-way trial. task_label_only uses no EEG.',
              'rows': len(keep), 'sentences': len(sentences), 'sentences_by_group': {g: int((groups == g).sum()) for g in sorted(set(groups))}, 'two_way': {}, 'n_way': {}}
    rng = np.random.default_rng(42)
    draws = [np.bincount(rng.integers(len(sentences), size=len(sentences)), minlength=len(sentences)) for _ in range(ITERATIONS)]
    for name, values in vectors.items():
        matrix = sentence_matrix(pair_wins(values, own, bank), own, len(sentences))
        within, across, overall = split_score(matrix, same)
        boot = np.array([split_score(matrix, same, w) for w in draws])
        result['two_way'][name] = {'overall': overall, 'same_task': within, 'different_task': across,
                                   'same_task_interval_95': np.percentile(boot[:, 0], [2.5, 97.5]).tolist(),
                                   'different_task_interval_95': np.percentile(boot[:, 1], [2.5, 97.5]).tolist()}
        entry = {'all_tasks': {f'{w}_way': distinct_trials(values, targets, sentence_rows, np.arange(len(sentences)), w) for w in WAYS}}
        for group in sorted(set(groups)):
            members = np.flatnonzero(groups == group)
            entry[group] = {f'{w}_way': distinct_trials(values, targets, sentence_rows, members, w) for w in WAYS if len(members) >= w}
        result['n_way'][name] = entry
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT/'scores.json', result)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
