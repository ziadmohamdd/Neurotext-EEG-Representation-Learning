"""Run proximity attribution test v3. The v2 test zone is never read, fitted or scored."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch
from eeg_decoding.data.dataset import read_manifest
from eeg_decoding.data.pipeline_view import PipelineView, ReaderNormalizer, pipeline_loader
from eeg_decoding.data.validation import manifest_digest
from eeg_decoding.evaluation.block_holdout import list_ranges, position_features
from eeg_decoding.evaluation.controls import shuffled_keys
from eeg_decoding.evaluation.proximity import (closest_text_prediction, excluded_positions, proximity_audit, proximity_decision,
                                               proximity_folds, rows_for)
from eeg_decoding.evaluation.sentence_holdout import report_scores, score_queries, word_count
from eeg_decoding.evaluation.spectral_pilot import count_basis
from eeg_decoding.models.classical import eeg_features, ridge_path
from eeg_decoding.models.eeg_cnn import EEGCNN
from eeg_decoding.models.projection import EEGTextModel
from eeg_decoding.training.alignment_train import TextCache, train_alignment
from eeg_decoding.training.checkpoints import load_checkpoint
from eeg_decoding.utils.config import load_config
from eeg_decoding.utils.device import device_info
from eeg_decoding.utils.logging import write_json
from eeg_decoding.utils.reproducibility import seed_all

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'configs/proximity_attribution_v3_20261105.json'
APPROVED_SHA256 = 'd6b03507fd34de056f5cadc57d16d04fb862cc966468559598f3f29126fabb8c'
PRIMARY = 10
SELECT_WAY = 4
WINDOW_SECONDS = {'F': 30, 'v1': 8}
# (input, condition, size, arm); near and far always run as a pair.
ARMS = [('F', c, 'full', a) for c in ('near', 'far') for a in ('real', 'gaussian', 'shuffled')] + \
       [('F', c, 'half', 'real') for c in ('near', 'far')] + [('v1', c, 'full', 'real') for c in ('near', 'far')]


class TimeLimitExceeded(RuntimeError):
    pass


class TimedLoader:
    """Stop an arm at its frozen wall-clock limit, between batches."""
    def __init__(self, inner, deadline):
        self.inner, self.deadline = inner, deadline

    def __len__(self):
        return len(self.inner)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __iter__(self):
        for batch in self.inner:
            if time.monotonic() > self.deadline:
                raise TimeLimitExceeded('Arm reached its frozen time limit')
            yield batch


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def ridge_predict(features, weights):
    return ((features - weights['mean']) / weights['scale']) @ weights['coef'].T + weights['intercept']


def timing_features(rows, training_counts):
    """Observation duration and word count only; no amplitudes enter this arm."""
    durations, counts = [], []
    for r in rows:
        with np.load(r.signal_path, allow_pickle=False) as f:
            durations.append(f['eeg'].shape[1] / r.sfreq)
        counts.append(word_count(r.text))
    counts = np.array(counts, dtype=float)
    basis, _ = count_basis(training_counts, counts)
    return np.c_[durations, counts, basis]


def feature_rows(rows, pipeline, normalizer, control, seed):
    view = PipelineView(rows, normalizer, pipeline, 'gaussian' if control == 'gaussian' else 'real', seed)
    first = eeg_features(view[0]['eeg'].numpy(), rows[0].sfreq)
    output = np.zeros((len(rows), len(first)), dtype=np.float32)
    output[0] = first
    for i in range(1, len(rows)):
        output[i] = eeg_features(view[i]['eeg'].numpy(), rows[i].sfreq)
    return output


def resolved_pipeline(plan, name, sfreq, cnn_config):
    samples = cnn_config.get('window_samples', int(WINDOW_SECONDS[name] * sfreq))
    return dict(plan['inputs'][name], window_samples=samples)


def score(predicted, rows, text, ways, matched_size):
    scored = score_queries(predicted, rows, text, ways, PRIMARY, matched_size)
    return report_scores(scored, rows, PRIMARY)


def selection_metric(predicted, rows, text):
    """Selection pools are too small for 10-way, so settings are chosen on 4-way."""
    return report_scores(score_queries(predicted, rows, text, (SELECT_WAY,), SELECT_WAY, SELECT_WAY), rows, SELECT_WAY)['metrics'][f'{SELECT_WAY}_way']


def select_ridge(train, targets, select, rows, text, alphas):
    """First grid entry wins ties, as the plan states."""
    best, chosen, table = -np.inf, None, []
    for alpha, weights in ridge_path(train, targets, alphas):
        metric = selection_metric(ridge_predict(select, weights), rows, text)
        table.append({'alpha': alpha, 'selection_metric': metric})
        if metric > best:
            best, chosen = metric, dict(weights, alpha=alpha)
    return chosen, table


def build_cnn(channels, config, text_dim):
    return EEGTextModel(EEGCNN(channels, config['embedding_dim'], quality_aware=False, temporal_segments=1),
                        config['embedding_dim'], text_dim)


def predict_cnn(model, rows, config, normalizer, pipeline, control, device):
    model.to(device).eval()
    output = []
    with torch.no_grad():
        for batch in pipeline_loader(rows, config, normalizer, pipeline, control='gaussian' if control == 'gaussian' else 'real'):
            output.append(model(batch['eeg'].to(device), batch['mask'].to(device)).float().cpu().numpy())
    return np.concatenate(output)


def run_attribution(records, text_path, cnn_config, plan, output, ways=(2, 4, 10), matched_size=20, iterations=10000, resume=False, status=print):
    """Train every condition per fold, score held-out clusters, then apply the frozen reading."""
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError('Use a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    seed, layout = plan['seed'], plan['split']
    excluded = excluded_positions(records, seed, **layout['v2_test_zone'])
    folds = proximity_folds(records, excluded, layout['cycle_sentences'], layout['cluster_sentences'], layout['fold_offsets'],
                            layout['near_lag'], layout['gap_lags'], seed)
    text = TextCache(text_path)
    leakage = proximity_audit(records, folds, excluded, text, layout['near_lag'], layout['gap_lags'], PRIMARY, SELECT_WAY)
    write_json(output/'leakage_audit.json', leakage)
    if leakage['failures']:
        raise ValueError('Leakage audit failed: ' + '; '.join(leakage['failures']))
    # Nothing past this point may touch a v2 test sentence, so the scope is cut here.
    forbidden_ids = {(release, i) for release, ids in excluded.items() for i in ids}
    allowed = [r for r in records if (r.dataset, int(r.sentence_id)) not in forbidden_ids]
    device, info = device_info(cnn_config.get('device', 'auto'))
    alphas = plan['runs']['ridge']['alphas']
    max_seconds = plan['runs']['cnn']['max_hours_per_arm'] * 3600
    ranges = list_ranges(records)
    sfreq = records[0].sfreq
    pipelines = {name: resolved_pipeline(plan, name, sfreq, cnn_config) for name in plan['inputs']}
    per_reader = {'F': True, 'v1': False}
    by_identity = {(r.dataset, int(r.sentence_id)): r.key for r in records}
    collected = {}

    def finished(folder):
        path = folder/'scored.json'
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None

    def record(folder, rows, predicted, **extra):
        folder.mkdir(parents=True, exist_ok=True)
        report = dict(score(predicted, rows, text, ways, matched_size), manifest_digest=manifest_digest(rows), **extra)
        write_json(folder/'scored.json', report)
        return report

    for fold, roles in enumerate(folds):
        place = output/f'fold{fold}'
        place.mkdir(exist_ok=True)
        select_rows, score_rows = rows_for(allowed, roles['select']), rows_for(allowed, roles['score'])
        write_json(place/'roles.json', {name: {k: sorted(v) for k, v in value.items()} for name, value in roles.items()})
        text_lookup = {identity: text.get([key])[0] for identity, key in by_identity.items()}
        for condition in ('near', 'far'):
            train_rows = rows_for(allowed, roles[condition])
            keys = [r.key for r in train_rows]
            targets = text.get(keys).astype(np.float64)
            counts = np.array([word_count(r.text) for r in train_rows], dtype=float)
            name = f'timing_{condition}'
            if not (report := finished(place/name)):
                status(f'RUN fold{fold} {name}')
                x, v = timing_features(train_rows, counts), timing_features(select_rows, counts)
                weights, table = select_ridge(x, targets, v, select_rows, text, alphas)
                report = record(place/name, score_rows, ridge_predict(timing_features(score_rows, counts), weights), selection=table)
            collected[fold, name] = report
            name = f'position_{condition}'
            if not (report := finished(place/name)):
                status(f'RUN fold{fold} {name}')
                spec = plan['runs']['no_eeg_controls']['position']
                best, chosen, grid = -np.inf, None, []
                for terms in spec['fourier_terms']:
                    x, v = position_features(train_rows, terms, ranges), position_features(select_rows, terms, ranges)
                    weights, table = select_ridge(x, targets, v, select_rows, text, spec['alphas'])
                    value = max(row['selection_metric'] for row in table)
                    grid.append({'fourier_terms': terms, 'grid': table})
                    if value > best:
                        best, chosen = value, (terms, weights)
                terms, weights = chosen
                report = record(place/name, score_rows, ridge_predict(position_features(score_rows, terms, ranges), weights), selection=grid, selected_terms=terms)
            collected[fold, name] = report
            name = f'text_{condition}'
            if not (report := finished(place/name)):
                status(f'RUN fold{fold} {name}')
                report = record(place/name, score_rows, closest_text_prediction(score_rows, roles[condition], text_lookup))
            collected[fold, name] = report
        for model_kind in ('ridge', 'cnn'):
            for pipeline_name, condition, size, control in ARMS:
                role = condition if size == 'full' else f'{condition}_half'
                name = f'{model_kind}_{pipeline_name}_{size}_{condition}_{control}'
                if (report := finished(place/name)):
                    collected[fold, name] = report
                    continue
                status(f'RUN fold{fold} {name}')
                train_rows = rows_for(allowed, roles[role])
                pipeline = pipelines[pipeline_name]
                normalizer = ReaderNormalizer().fit(train_rows, pipeline, per_reader=per_reader[pipeline_name])
                if model_kind == 'ridge':
                    x = feature_rows(train_rows, pipeline, normalizer, control, seed)
                    v = feature_rows(select_rows, pipeline, normalizer, control, seed)
                    keys = [r.key for r in train_rows]
                    keys = shuffled_keys(keys, seed) if control == 'shuffled' else keys
                    weights, table = select_ridge(x, text.get(keys).astype(np.float64), v, select_rows, text, alphas)
                    predicted = ridge_predict(feature_rows(score_rows, pipeline, normalizer, control, seed), weights)
                    collected[fold, name] = record(place/name, score_rows, predicted, selection=table, selected_alpha=float(weights['alpha']))
                    continue
                model_seed = plan['runs']['cnn']['model_seed_by_fold'][fold]
                config = dict(cnn_config, seed=model_seed)
                seed_all(model_seed)
                channels = len(train_rows[0].channels) - len(pipeline['drop_channels'])
                model = build_cnn(channels, config, text.embeddings.shape[1])
                folder = place/name
                folder.mkdir(exist_ok=True)
                deadline = time.monotonic() + max_seconds
                train = TimedLoader(pipeline_loader(train_rows, config, normalizer, pipeline, True, True, control), deadline)
                select = TimedLoader(pipeline_loader(select_rows, config, normalizer, pipeline, False, True,
                                                     'gaussian' if control == 'gaussian' else 'real'), deadline)
                metadata = dict(config=config, stage='alignment', pipeline=pipeline, training_control=control,
                                split_digest=manifest_digest(train_rows), text_cache_digest=text.digest, device_info=info,
                                encoder_origin='supervised_cnn')
                last = folder/'last.pt'
                started = time.monotonic()
                history = train_alignment(model, train, select, text, config, device, folder, metadata, last if last.exists() else None)
                load_checkpoint(last, model)
                final = score(predict_cnn(model, score_rows, config, normalizer, pipeline, control, device), score_rows, text, ways, matched_size)
                best = load_checkpoint(folder/'best.pt', model)
                predicted = predict_cnn(model, score_rows, config, normalizer, pipeline, control, device)
                collected[fold, name] = record(folder, score_rows, predicted, best_epoch=best['epoch'], epochs_run=len(history),
                                               selection_loss=best['best'], seconds=time.monotonic()-started,
                                               last_epoch={'metrics': final['metrics'], 'primary': [q['primary'] for q in final['queries']]})
                del model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

    # Each sentence is scored in one fold, so joining folds keeps every query once.
    joined_rows = [r for roles in folds for r in rows_for(allowed, roles['score'])]

    def values(name, field=None):
        output_values = []
        for fold in range(len(folds)):
            report = collected[fold, name]
            output_values += report['last_epoch']['primary'] if field == 'last' else [q['primary'] for q in report['queries']]
        return np.array(output_values)

    def paired(prefix, control='real', field=None):
        return values(f'{prefix}_near_{control}', field), values(f'{prefix}_far_{control}', field)

    def macro(v):
        readers = sorted({r.subject_key for r in joined_rows})
        return float(np.mean([v[[r.subject_key == s for r in joined_rows]].mean() for s in readers]))

    decision = {}
    for model_kind in ('cnn', 'ridge'):
        prefix = f'{model_kind}_F_full'
        near, far = paired(prefix)
        decision[model_kind] = proximity_decision(near, far, values(f'{prefix}_far_gaussian'), joined_rows, iterations)
        decision[model_kind]['size_effect'] = {c: macro(values(f'{prefix}_{c}_real') - values(f'{model_kind}_F_half_{c}_real')) for c in ('near', 'far')}
        decision[model_kind]['input_effect_v1_minus_F'] = {c: macro(values(f'{model_kind}_v1_full_{c}_real') - values(f'{prefix}_{c}_real')) for c in ('near', 'far')}
        decision[model_kind]['controls'] = {f'{control}_{c}': macro(values(f'{prefix}_{c}_{control}')) for control in ('gaussian', 'shuffled') for c in ('near', 'far')}
    near, far = paired('cnn_F_full', field='last')
    decision['cnn']['last_epoch'] = {'near': macro(near), 'far': macro(far), 'gain': macro(near - far)}
    decision['no_eeg'] = {name: {c: macro(values(f'{name}_{c}')) for c in ('near', 'far')} for name in ('timing', 'position', 'text')}
    decision['scored_observations'] = len(joined_rows)
    write_json(output/'decision.json', decision)
    return collected, decision


def main(output, resume):
    if sha(PLAN) != APPROVED_SHA256:
        raise ValueError('Frozen plan changed; a change needs a new version')
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    manifests = [ROOT/p for p in plan['cohort']['manifests']]
    text_path = ROOT/plan['text_cache']
    records = [r for path in manifests for r in read_manifest(path)]
    if len(records) != plan['cohort']['observations'] or len({r.subject_key for r in records}) != plan['cohort']['readers']:
        raise ValueError('Cohort differs from the frozen plan')
    if output.exists() and not resume:
        raise FileExistsError('Use a fresh output directory')
    cnn = plan['runs']['cnn']
    config = dict(load_config(ROOT/'configs/local.yaml'), seed=plan['seed'], embedding_dim=cnn['embedding_dim'],
                  batch_size=cnn['batch_size'], gradient_accumulation_steps=cnn['gradient_accumulation_steps'],
                  alignment_candidate_scope=cnn['alignment_candidate_scope'], learning_rate=cnn['learning_rate'],
                  weight_decay=cnn['weight_decay'], gradient_clip=cnn['gradient_clip'], epochs=cnn['max_epochs'],
                  patience=cnn['patience'], temporal_segments=1, quality_aware=False)
    prior = sorted(p for p in (ROOT/'results').rglob('*') if p.is_file() and output.resolve() not in p.resolve().parents)
    protected = prior + manifests + [text_path] + sorted({Path(r.signal_path) for r in records})
    before_path = output/'preservation_before.json'
    output.mkdir(parents=True, exist_ok=resume)
    if not before_path.exists():
        print('Hashing protected files', flush=True)
        write_json(before_path, {str(p): sha(p) for p in protected})
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        write_json(output/'plan.json', dict(plan_sha256=APPROVED_SHA256, plan=plan, config=config, git_head=commit))
        (output/'implementation').mkdir(exist_ok=True)
        for source in [PLAN, ROOT/'docs/proximity_attribution_plan_v3_20261105.md', Path(__file__), ROOT/'src/eeg_decoding/evaluation/proximity.py',
                       ROOT/'tests/test_proximity.py']:
            shutil.copyfile(source, output/'implementation'/source.name)
    _, decision = run_attribution(records, text_path, config, plan, output, resume=True, status=lambda m: print(m, flush=True))
    before = json.loads(before_path.read_text(encoding='utf-8'))
    after = {p: sha(p) for p in before}
    if after != before:
        raise RuntimeError('A protected file changed')
    write_json(output/'preservation_after.json', {'unchanged': True, 'protected_files': len(before)})
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/proximity_attribution_v3_20261105')
    parser.add_argument('--resume', action='store_true')
    arguments = parser.parse_args()
    main(arguments.output, arguments.resume)
