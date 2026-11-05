"""Run position bridge test v4. v1 test sentences and the v2 test zone are never fitted or scored."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from scipy.stats import spearmanr
import torch
from eeg_decoding.data.dataset import read_manifest
from eeg_decoding.data.pipeline_view import PipelineView, ReaderNormalizer
from eeg_decoding.evaluation.block_holdout import _macro, bootstrap_gap
from eeg_decoding.evaluation.nway import score_counts
from eeg_decoding.evaluation.position_bridge import (bridge_prediction, bridge_reading, bridge_scope, fit_positions, identity, missing_runs,
                                                     position_quality, predict_positions, random_split_baseline, scope_audit,
                                                     shuffled_position_map)
from eeg_decoding.evaluation.proximity import closest_text_prediction, excluded_positions
from eeg_decoding.evaluation.retrieval import normalize
from eeg_decoding.evaluation.sentence_holdout import report_scores, score_queries, sentence_holdout_split
from eeg_decoding.models.classical import eeg_features
from eeg_decoding.models.eeg_cnn import EEGCNN
from eeg_decoding.models.projection import EEGTextModel
from eeg_decoding.preprocessing.normalization import ChannelNormalizer
from eeg_decoding.training.alignment_train import TextCache
from eeg_decoding.training.checkpoints import load_checkpoint
from eeg_decoding.utils.device import device_info
from eeg_decoding.utils.logging import write_json
from eeg_decoding.workflow import loader

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'configs/position_bridge_v4_20261105.json'
APPROVED_SHA256 = '203d9d25d3a1ad75cf18fc2ecac50f0598c89d51ee8f233c39ab71fea5318d8e'
V1 = ROOT/'results/sentence_holdout_pilot_v1_20261105'
PRIMARY = 10
WAYS = (2, 4, 10, 24)
MATCHED = 20
WINDOW_SECONDS = {'F': 30, 'v1': 8}
PER_READER = {'F': True, 'v1': False}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def feature_rows(rows, pipeline, normalizer, control, seed):
    view = PipelineView(rows, normalizer, pipeline, control, seed)
    first = eeg_features(view[0]['eeg'].numpy(), rows[0].sfreq)
    output = np.zeros((len(rows), len(first)), dtype=np.float32)
    output[0] = first
    for i in range(1, len(rows)):
        output[i] = eeg_features(view[i]['eeg'].numpy(), rows[i].sfreq)
    return output


def score(predicted, rows, text):
    return score_queries(predicted, rows, text, WAYS, PRIMARY, MATCHED)


def v1_cnn_prediction(rows):
    """Re-predict the saved v1 CNN on the kept queries only. Its stored normalizer is reused, so no other EEG is read."""
    state = torch.load(V1/'cnn_real/best.pt', map_location='cpu', weights_only=True)
    meta = state['metadata']
    config = meta['config']
    device, _ = device_info(config.get('device', 'auto'))
    model = EEGTextModel(EEGCNN(len(meta['channels']), config['embedding_dim'], quality_aware=False, temporal_segments=1),
                         config['embedding_dim'], 384)
    load_checkpoint(V1/'cnn_real/best.pt', model)
    model.to(device).eval()
    norm = ChannelNormalizer.from_state(meta['normalizer'])
    output = []
    with torch.no_grad():
        for batch in loader(rows, config, norm, control='real'):
            output.append(model(batch['eeg'].to(device), batch['mask'].to(device)).float().cpu().numpy())
    return np.concatenate(output)


def verify_saved_ranks(predicted, rows, all_val_rows, text):
    """Rank the re-predicted queries against the original v1 pool, using text only, and compare with the saved ranks."""
    saved = {tuple(q['identity']): q['rank'] for q in json.loads((V1/'cnn_real/validation.json').read_text(encoding='utf-8'))['queries']}
    predicted = normalize(predicted)
    differences = []
    for release in sorted({r.dataset for r in rows}):
        pool = sorted({r.key for r in all_val_rows if r.dataset == release})
        lookup = {k: i for i, k in enumerate(pool)}
        mine = [i for i, r in enumerate(rows) if r.dataset == release]
        scores = predicted[mine] @ normalize(text.get(pool)).T
        _, _, rank = score_counts(scores, np.array([lookup[rows[i].key] for i in mine]))
        for i, value in zip(mine, rank):
            r = rows[i]
            differences.append(abs(float(value) - saved[r.dataset, r.subject, r.session, r.task, r.sentence_id]))
    differences = np.array(differences)
    return {'queries': len(differences), 'share_identical_rank': float(np.mean(differences == 0)), 'max_rank_difference': float(differences.max())}


def interval_of(values, rows, iterations):
    report = bootstrap_gap(values, rows, iterations, 42, level=95)
    return {'estimate': _macro(values, rows), 'reader_interval': report['reader_interval'], 'sentence_interval': report['sentence_interval']}


def run_bridge(records, text_path, plan, output, reference=v1_cnn_prediction, verify=verify_saved_ranks, iterations=10000, fractions=None, seeds=None,
               zone=None, status=print):
    output = Path(output)
    seed, spec = plan['seed'], plan['bridge']
    zone = zone or plan['protected']['v2_test_zone']
    closed_by_release = excluded_positions(records, seed, **zone)
    train_rows, rows, closed, v1_test = bridge_scope(records, seed, closed_by_release)
    all_val_rows = sentence_holdout_split(records, seed)['val']
    audit = scope_audit(train_rows, rows, closed, v1_test)
    write_json(output/'leakage_audit.json', audit)
    if audit['failures']:
        raise ValueError('Scope audit failed: ' + '; '.join(audit['failures']))
    text = TextCache(text_path)
    by_identity = {identity(r): r.key for r in records}
    text_by_identity = {i: text.get([k])[0] for i, k in by_identity.items()}
    train_ids = {}
    for r in train_rows:
        train_ids.setdefault(r.dataset, set()).add(int(r.sentence_id))
    results, values = {}, {}

    def report(name, predicted, **extra):
        scored = score(predicted, rows, text)
        values[name] = scored['expected'][PRIMARY]
        results[name] = dict(report_scores(scored, rows, PRIMARY)['metrics'], **extra)
        status(f'{name}: 10-way {results[name][f"{PRIMARY}_way"]:.4f}')

    status('RUN v1 CNN on kept queries')
    predicted = reference(rows)
    check = verify(predicted, rows, all_val_rows, text)
    if check['share_identical_rank'] < plan['reference']['minimum_identical_rank_share']:
        raise ValueError(f'Re-predicted CNN differs from the saved run: {check}')
    report('v1_cnn', predicted, saved_rank_check=check)
    report('neighbour_text', closest_text_prediction(rows, train_ids, text_by_identity, spec['neighbours']))

    for name, pipeline_spec in plan['inputs'].items():
        pipeline = dict(pipeline_spec, window_samples=int(WINDOW_SECONDS[name] * records[0].sfreq))
        status(f'FIT normalizer {name}')
        normalizer = ReaderNormalizer().fit(train_rows, pipeline, per_reader=PER_READER[name])
        for control in ('real', 'gaussian'):
            status(f'FEATURES {name} {control}')
            x, v = feature_rows(train_rows, pipeline, normalizer, control, seed), feature_rows(rows, pipeline, normalizer, control, seed)
            models, tables = fit_positions(x, train_rows, spec['alphas'], seed)
            positions = predict_positions(models, v, rows)
            report(f'bridge_{name}_{control}', bridge_prediction(positions, rows, train_ids, text_by_identity, spec['neighbours']),
                   position=position_quality(positions, rows), selection=tables)
            if control == 'real':
                stacked = []
                for shuffle in range(spec['shuffles']):
                    mapping = shuffled_position_map(train_rows, seed + shuffle)
                    models, _ = fit_positions(x, train_rows, spec['alphas'], seed, mapping)
                    positions = predict_positions(models, v, rows)
                    stacked.append(score(bridge_prediction(positions, rows, train_ids, text_by_identity, spec['neighbours']), rows, text)['expected'])
                values[f'bridge_{name}_shuffled'] = np.mean([s[PRIMARY] for s in stacked], axis=0)
                results[f'bridge_{name}_shuffled'] = {f'{w}_way': _macro(np.mean([s[w] for s in stacked], axis=0), rows) for w in WAYS}
                status(f'bridge_{name}_shuffled: 10-way {results[f"bridge_{name}_shuffled"][f"{PRIMARY}_way"]:.4f}')

    primary = plan['evaluation']['primary_input']
    decision = bridge_reading(values[f'bridge_{primary}_real'], values[f'bridge_{primary}_shuffled'], results['v1_cnn'][f'{PRIMARY}_way'], rows, iterations)
    decision['secondary_input'] = {name: bridge_reading(values[f'bridge_{name}_real'], values[f'bridge_{name}_shuffled'], results['v1_cnn'][f'{PRIMARY}_way'], rows, iterations)
                                   for name in plan['inputs'] if name != primary}
    decision['real_minus_noise'] = {name: interval_of(values[f'bridge_{name}_real'] - values[f'bridge_{name}_gaussian'], rows, iterations) for name in plan['inputs']}
    decision['cnn_minus_bridge'] = interval_of(values['v1_cnn'] - values[f'bridge_{primary}_real'], rows, iterations)
    sentences = sorted({r.key for r in rows})

    def by_sentence(name):
        return [float(np.mean([v for v, r in zip(values[name], rows) if r.key == key])) for key in sentences]

    correlation = spearmanr(by_sentence('v1_cnn'), by_sentence(f'bridge_{primary}_real'))
    decision['sentence_spearman_cnn_bridge'] = {'rho': float(correlation[0]), 'p': float(correlation[1]), 'sentences': len(sentences)}
    status('RUN random split baseline')
    random_splits = random_split_baseline(records, closed, v1_test, text_by_identity, text, fractions or plan['random_splits']['fractions'],
                                          seeds or range(plan['random_splits']['seeds']))
    runs = missing_runs(records)
    write_json(output/'scores.json', results)
    write_json(output/'decision.json', decision)
    write_json(output/'random_splits.json', random_splits)
    write_json(output/'sentence_id_check.json', {'source': 'matlab_sentence_index minus one, per reader file', 'missing_runs': runs})
    np.savez_compressed(output/'query_scores.npz', **values)
    return results, decision, random_splits


def main(output):
    if sha(PLAN) != APPROVED_SHA256:
        raise ValueError('Frozen plan changed; a change needs a new version')
    if output.exists():
        raise FileExistsError('Use a fresh output directory')
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    manifests = [ROOT/p for p in plan['cohort']['manifests']]
    text_path = ROOT/plan['text_cache']
    records = [r for path in manifests for r in read_manifest(path)]
    if len(records) != plan['cohort']['observations'] or len({r.subject_key for r in records}) != plan['cohort']['readers']:
        raise ValueError('Cohort differs from the frozen plan')
    prior = sorted(p for p in (ROOT/'results').rglob('*') if p.is_file())
    protected = prior + manifests + [text_path] + sorted({Path(r.signal_path) for r in records})
    output.mkdir(parents=True)
    print('Hashing protected files', flush=True)
    before = {str(p): sha(p) for p in protected}
    write_json(output/'preservation_before.json', before)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    write_json(output/'plan.json', dict(plan_sha256=APPROVED_SHA256, plan=plan, git_head=commit))
    (output/'implementation').mkdir()
    for source in [PLAN, ROOT/'docs/position_bridge_plan_v4_20261105.md', Path(__file__), ROOT/'src/eeg_decoding/evaluation/position_bridge.py', ROOT/'tests/test_position_bridge.py']:
        shutil.copyfile(source, output/'implementation'/source.name)
    _, decision, _ = run_bridge(records, text_path, plan, output, status=lambda m: print(m, flush=True))
    if {p: sha(p) for p in before} != before:
        raise RuntimeError('A protected file changed')
    write_json(output/'preservation_after.json', {'unchanged': True, 'protected_files': len(before)})
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/position_bridge_v4_20261105')
    main(parser.parse_args().output)
