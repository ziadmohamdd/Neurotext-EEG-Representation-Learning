"""Score the v1 validation split from the text of neighbouring training sentences. No EEG, no training, no test sentence scored."""
import json
from pathlib import Path
import numpy as np
from eeg_decoding.data.dataset import read_manifest
from eeg_decoding.evaluation.proximity import closest_text_prediction
from eeg_decoding.evaluation.sentence_holdout import report_scores, score_queries, sentence_holdout_split
from eeg_decoding.training.alignment_train import TextCache
from eeg_decoding.utils.logging import write_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'results/sentence_holdout_pilot_v1_20261105'
OUTPUT = ROOT/'results/v1_neighbour_text_20261105'
WAYS = (2, 4, 10, 24)


def main():
    if OUTPUT.exists():
        raise FileExistsError('Use a fresh output directory')
    plan = json.loads((SOURCE/'plan.json').read_text(encoding='utf-8'))['plan']
    records = [r for path in plan['cohort']['manifests'] for r in read_manifest(ROOT/path)]
    splits = sentence_holdout_split(records, plan['seed'])
    text = TextCache(ROOT/plan['text_cache'])
    train_ids, lookup = {}, {}
    for r in splits['train']:
        train_ids.setdefault(r.dataset, set()).add(int(r.sentence_id))
        lookup[r.dataset, int(r.sentence_id)] = text.get([r.key])[0]
    rows = splits['val']
    saved = json.loads((SOURCE/'cnn_real/validation.json').read_text(encoding='utf-8'))
    if [str(r.sentence_id) for r in rows] != [str(q['identity'][4]) for q in saved['queries']]:
        raise ValueError('Rows do not line up with the saved v1 validation queries')
    result = {'note': 'Added after v3. Reuses v1 validation sentences. Pool and scoring match the v1 CNN.', 'observations': len(rows), 'cnn_real_v1': saved['metrics']}
    for count in (1, 2):
        predicted = closest_text_prediction(rows, train_ids, lookup, count)
        result[f'closest_{count}'] = report_scores(score_queries(predicted, rows, text, WAYS, 10, 20), rows, 10)['metrics']
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT/'scores.json', result)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
