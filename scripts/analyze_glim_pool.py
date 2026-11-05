"""Score GLIM's published outputs across all tasks and within each task, leaving out protected sentences. No EEG; nothing is trained.

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
OUTPUT = ROOT/'results/glim_pool_check_20261106'
FILES = {'glim': 'wandb_export_gen_samples_glim.csv', 'noise_input': 'wandb_export_gen_samples_noise_input_test.csv',
         'prompt_free': 'wandb_export_gen_samples_prompt_free_test.csv'}
MODEL, REVISION = 'sentence-transformers/all-MiniLM-L6-v2', '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
WAYS = (2, 4, 10, 24)
TRIALS = 10000
csv.field_size_limit(10**8)


def squash(sentence):
    return re.sub(r'[^a-z0-9]+', '', sentence.lower())


def read_rows(path):
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def trials(predicted, targets, seed=42):
    output = {}
    for way in WAYS:
        rng = np.random.default_rng(seed)
        total = 0.
        for _ in range(TRIALS):
            picks = rng.choice(len(predicted), way, replace=False)
            total += np.mean((predicted[picks] @ targets[picks].T).argmax(1) == np.arange(way))
        output[f'{way}_way'] = total / TRIALS
    return output


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

    targets = encode([base[i]['Input Text'] for i in keep])
    vectors = {name: encode([table[i]['Generated Text'] for i in keep]) for name, table in outputs.items()}
    groups = {'all_tasks': np.arange(len(keep))}
    labels = [f"{base[i]['Dataset']}_{base[i]['Raw Task Key']}" for i in keep]
    for label in sorted(set(labels)):
        groups[label] = np.array([j for j, value in enumerate(labels) if value == label])
    result = {'note': 'Exploratory. Protected sentences removed by text match. Rows from tasks not held locally cannot be checked against the list.',
              'rows_total': len(base), 'rows_kept': len(keep), 'groups': {}}
    for label, chosen in groups.items():
        entry = {'rows': len(chosen), 'sentences': len({base[keep[j]]['Input Text'] for j in chosen})}
        for name, values in vectors.items():
            entry[name] = trials(values[chosen], targets[chosen])
        result['groups'][label] = entry
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT/'scores.json', result)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
