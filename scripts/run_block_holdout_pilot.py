"""Run block-holdout pilot v2. Test blocks are read only after a validation gate pass."""
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
from eeg_decoding.data.pipeline_view import PipelineView, ReaderNormalizer, center_crop, clean_signal, pipeline_loader
from eeg_decoding.data.validation import manifest_digest
from eeg_decoding.evaluation.block_holdout import (apply_block_gate, block_holdout_split, leakage_audit, list_ranges,
                                                    position_features, within_block_scores)
from eeg_decoding.evaluation.controls import shuffled_keys
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
PLAN = ROOT/'configs/block_holdout_pilot_v2_20261105.json'
APPROVED_SHA256 = 'eef247790d718091b5f3e5357feb1989ceb7a3881f852868170ad89a641f9006'
PRIMARY = 10
CONTROLS = ('real', 'gaussian', 'shuffled')
WINDOW_SECONDS = {'V': 8, 'F': 30}


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


def resolved_pipeline(plan, name, sfreq):
    return dict(plan['pipelines'][name], window_samples=int(WINDOW_SECONDS[name] * sfreq))


class Scorer:
    def __init__(self, text, blocks, ways, matched_size, within_way):
        self.text, self.blocks, self.ways, self.matched_size, self.within_way = text, blocks, ways, matched_size, within_way

    def primary(self, predicted, rows):
        return score_queries(predicted, rows, self.text, self.ways, PRIMARY, self.matched_size)

    def full(self, predicted, rows):
        scored = self.primary(predicted, rows)
        within = within_block_scores(predicted, rows, self.blocks, self.text, self.within_way)
        report = report_scores(scored, rows, PRIMARY)
        report['metrics']['within_block_%d_way' % self.within_way] = float(
            np.mean([within[[r.subject_key == s for r in rows]].mean() for s in sorted({r.subject_key for r in rows})]))
        for query, value in zip(report['queries'], within):
            query['within_block'] = float(value)
        return report


def select_ridge(train, targets, val, rows, scorer, alphas):
    """First grid entry wins ties, as the plan states."""
    best, chosen, table = -np.inf, None, []
    for alpha, weights in ridge_path(train, targets, alphas):
        report = report_scores(scorer.primary(ridge_predict(val, weights), rows), rows, PRIMARY)
        metric = report['metrics'][f'{PRIMARY}_way']
        table.append({'alpha': alpha, 'validation_primary': metric})
        if metric > best:
            best, chosen = metric, dict(weights, alpha=alpha)
    return chosen, table


def position_correlation(train, train_rows, val, val_rows):
    """Secondary check of how well features recover list position within each release."""
    output = {}
    for release in sorted({r.dataset for r in val_rows}):
        i = [k for k, r in enumerate(train_rows) if r.dataset == release]
        j = [k for k, r in enumerate(val_rows) if r.dataset == release]
        y = np.array([[float(train_rows[k].sentence_id)] for k in i])
        for _, weights in ridge_path(train[i], y, [1000.]):
            predicted = ridge_predict(val[j], weights).ravel()
        output[release] = float(np.corrcoef(predicted, [float(val_rows[k].sentence_id) for k in j])[0, 1])
    return output


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


def run_pilot(records, text_path, cnn_config, plan, output, ways=(2, 4, 10, 24), matched_size=20, within_way=10,
              readers_positive=20, iterations=10000, resume=False, status=print):
    """Validation arms, then the gate, then one test evaluation per passing model."""
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError('Use a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    seed, layout = plan['seed'], plan['split']
    splits, blocks, audit = block_holdout_split(records, seed, layout['block_sentences'], layout['buffer_sentences_each_side'],
                                                layout['blocks_per_release']['val'], layout['blocks_per_release']['test'])
    train_rows, val_rows = splits['train'], splits['val']
    for name in ('val', 'test'):
        if not {r.subject_key for r in splits[name]} <= {r.subject_key for r in train_rows}:
            raise ValueError('Every held-out reader must appear in training')
    text = TextCache(text_path)
    leakage = leakage_audit(splits, blocks, records, text, layout['buffer_sentences_each_side'])
    write_json(output/'leakage_audit.json', leakage)
    if leakage['failures']:
        raise ValueError('Leakage audit failed: ' + '; '.join(leakage['failures']))
    scorer = Scorer(text, blocks, ways, matched_size, within_way)
    write_json(output/'splits.json', {'audit': audit, 'digests': {k: manifest_digest(v) for k, v in splits.items()},
                                      'sentence_keys': {k: sorted({r.key for r in v}) for k, v in splits.items()},
                                      'blocks': sorted({*blocks.values()})})
    device, info = device_info(cnn_config.get('device', 'auto'))
    alphas = plan['models']['ridge']['alphas']
    max_seconds = plan['models']['cnn']['max_hours_per_arm'] * 3600
    train_keys = [r.key for r in train_rows]
    train_counts = np.array([word_count(r.text) for r in train_rows], dtype=float)
    ranges = list_ranges(records)
    sfreq = train_rows[0].sfreq
    pipelines = {name: resolved_pipeline(plan, name, sfreq) for name in plan['pipelines']}
    if 'window_samples' in cnn_config:
        pipelines = {k: dict(v, window_samples=cnn_config['window_samples']) for k, v in pipelines.items()}
    status('FIT normalizers')
    normalizers = {name: ReaderNormalizer().fit(train_rows, pipelines[name], per_reader=name == 'F') for name in pipelines}

    def finished(folder):
        path = output/folder/'validation.json'
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None

    def record(folder, rows, predicted, split, **extra):
        name = 'validation.json' if split == 'val' else 'test.json'
        if split == 'test' and (output/folder/name).exists():
            raise FileExistsError('Test blocks may be scored only once')
        (output/folder).mkdir(exist_ok=True)
        report = dict(scorer.full(predicted, rows), evaluation_split=split, manifest_digest=manifest_digest(rows), **extra)
        write_json(output/folder/name, report)
        return report

    reports = {}
    target_vectors = text.get(train_keys).astype(np.float64)
    reports['timing'] = finished('timing')
    if not reports['timing']:
        status('RUN timing')
        (output/'timing').mkdir(exist_ok=True)
        x, v = timing_features(train_rows, train_counts), timing_features(val_rows, train_counts)
        weights, table = select_ridge(x, target_vectors, v, val_rows, scorer, alphas)
        np.savez_compressed(output/'timing/ridge.npz', **weights)
        reports['timing'] = record('timing', val_rows, ridge_predict(v, weights), 'val', selection=table, selected_alpha=float(weights['alpha']))
    reports['position'] = finished('position')
    if not reports['position']:
        status('RUN position')
        (output/'position').mkdir(exist_ok=True)
        best, chosen, table = -np.inf, None, []
        spec = plan['shared_controls']['position_only_ridge']
        for terms in spec['fourier_terms']:
            x, v = position_features(train_rows, terms, ranges), position_features(val_rows, terms, ranges)
            weights, grid = select_ridge(x, target_vectors, v, val_rows, scorer, spec['alphas'])
            value = max(row['validation_primary'] for row in grid)
            table.append({'fourier_terms': terms, 'grid': grid})
            if value > best:
                best, chosen = value, (terms, weights)
        terms, weights = chosen
        np.savez_compressed(output/'position/ridge.npz', fourier_terms=terms, **weights)
        reports['position'] = record('position', val_rows, ridge_predict(position_features(val_rows, terms, ranges), weights), 'val',
                                     selection=table, selected_terms=terms, selected_alpha=float(weights['alpha']))
    for name in plan['models']['ridge']['pipelines']:
        for control in CONTROLS:
            folder = f'ridge_{name}_{control}'
            reports[folder] = finished(folder)
            if reports[folder]:
                continue
            status('RUN '+folder)
            (output/folder).mkdir(exist_ok=True)
            x = feature_rows(train_rows, pipelines[name], normalizers[name], control, seed)
            v = feature_rows(val_rows, pipelines[name], normalizers[name], control, seed)
            keys = shuffled_keys(train_keys, seed) if control == 'shuffled' else train_keys
            weights, table = select_ridge(x, text.get(keys).astype(np.float64), v, val_rows, scorer, alphas)
            np.savez_compressed(output/folder/'ridge.npz', **weights)
            reports[folder] = record(folder, val_rows, ridge_predict(v, weights), 'val', selection=table, selected_alpha=float(weights['alpha']),
                                     position_correlation=position_correlation(x, train_rows, v, val_rows))
    channels = len(train_rows[0].channels) - len(plan['pipelines']['F']['drop_channels'])
    model_seeds = plan['models']['cnn']['model_seeds']
    for name in plan['models']['cnn']['pipelines']:
        for control in CONTROLS:
            for model_seed in model_seeds:
                folder = f'cnn_{name}_{control}_s{model_seed}'
                reports[folder] = finished(folder)
                if reports[folder]:
                    continue
                status('RUN '+folder)
                (output/folder).mkdir(exist_ok=True)
                config = dict(cnn_config, seed=model_seed)
                seed_all(model_seed)
                model = build_cnn(channels, config, text.embeddings.shape[1])
                deadline = time.monotonic() + max_seconds
                train = TimedLoader(pipeline_loader(train_rows, config, normalizers[name], pipelines[name], True, True, control), deadline)
                val = TimedLoader(pipeline_loader(val_rows, config, normalizers[name], pipelines[name], False, True,
                                                  'gaussian' if control == 'gaussian' else 'real'), deadline)
                metadata = dict(config=config, stage='alignment', pipeline=pipelines[name], training_control=control,
                                split_digest=manifest_digest(train_rows), text_cache_digest=text.digest, device_info=info,
                                encoder_origin='supervised_cnn')
                last = output/folder/'last.pt'
                started = time.monotonic()
                history = train_alignment(model, train, val, text, config, device, output/folder, metadata, last if last.exists() else None)
                best = load_checkpoint(output/folder/'best.pt', model)
                predicted = predict_cnn(model, val_rows, config, normalizers[name], pipelines[name], control, device)
                reports[folder] = record(folder, val_rows, predicted, 'val', best_epoch=best['epoch'], epochs_run=len(history),
                                         validation_loss=best['best'], seconds=time.monotonic()-started)
                del model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

    def values(folder_names, field):
        return np.mean([[q[field] for q in reports[f]['queries']] for f in folder_names], axis=0)

    families = {f'ridge_{n}': {c: [f'ridge_{n}_{c}'] for c in CONTROLS} for n in plan['models']['ridge']['pipelines']}
    families.update({f'cnn_{n}': {c: [f'cnn_{n}_{c}_s{s}' for s in model_seeds] for c in CONTROLS} for n in plan['models']['cnn']['pipelines']})
    decision = {}
    for family, arms in families.items():
        comparators = {'gaussian': values(arms['gaussian'], 'primary'), 'class_shuffled': values(arms['shuffled'], 'primary'),
                       'timing_only_ridge': values(['timing'], 'primary'), 'position_only_ridge': values(['position'], 'primary')}
        decision[family] = apply_block_gate(values(arms['real'], 'primary'), comparators, values(arms['real'], 'within_block'),
                                            values(arms['gaussian'], 'within_block'), val_rows,
                                            readers_positive=readers_positive, iterations=iterations)
        decision[family]['real_primary'] = report_macro(values(arms['real'], 'primary'), val_rows)
    write_json(output/'decision.json', decision)
    # The gate is settled above. Only now may test rows be read, and only for a pass.
    tests = {}
    test_rows = splits['test']
    for family, result in decision.items():
        if not result['passed']:
            continue
        status('TEST '+family)
        if 'timing' not in tests:
            with np.load(output/'timing/ridge.npz', allow_pickle=False) as f:
                tests['timing'] = record('timing', test_rows, ridge_predict(timing_features(test_rows, train_counts), dict(f)), 'test')
            with np.load(output/'position/ridge.npz', allow_pickle=False) as f:
                saved = {k: f[k] for k in f.files}
            terms = int(saved.pop('fourier_terms'))
            tests['position'] = record('position', test_rows, ridge_predict(position_features(test_rows, terms, ranges), saved), 'test')
        name = family.split('_')[1]
        for control, folders in families[family].items():
            for folder in folders:
                if family.startswith('ridge'):
                    with np.load(output/folder/'ridge.npz', allow_pickle=False) as f:
                        predicted = ridge_predict(feature_rows(test_rows, pipelines[name], normalizers[name], control, seed), dict(f))
                else:
                    model_seed = int(folder.rsplit('_s', 1)[1])
                    config = dict(cnn_config, seed=model_seed)
                    model = build_cnn(channels, config, text.embeddings.shape[1])
                    load_checkpoint(output/folder/'best.pt', model)
                    predicted = predict_cnn(model, test_rows, config, normalizers[name], pipelines[name], control, device)
                tests[folder] = record(folder, test_rows, predicted, 'test')
    return reports, decision, tests


def report_macro(values, rows):
    readers = sorted({r.subject_key for r in rows})
    return float(np.mean([values[[r.subject_key == s for r in rows]].mean() for s in readers]))


def main(output, resume):
    if sha(PLAN) != APPROVED_SHA256:
        raise ValueError('Frozen plan changed; a change needs a new version')
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    manifests = [ROOT/p for p in plan['cohort']['manifests']]
    text_path = ROOT/plan['text_cache']
    records = [r for path in manifests for r in read_manifest(path)]
    if len(records) != plan['cohort']['observations'] or len({r.subject_key for r in records}) != plan['cohort']['readers']:
        raise ValueError('Cohort differs from the frozen plan')
    layout = plan['split']
    splits, _, _ = block_holdout_split(records, plan['seed'], layout['block_sentences'], layout['buffer_sentences_each_side'],
                                       layout['blocks_per_release']['val'], layout['blocks_per_release']['test'])
    for name, rows in splits.items():
        if len(rows) != layout['exact_sizes'][name]['observations']:
            raise ValueError(f'Split {name} differs from the frozen plan')
    if output.exists() and not resume:
        raise FileExistsError('Use a fresh output directory')
    cnn = plan['models']['cnn']
    if 'embedding_dim 128' not in cnn['encoder']:
        raise ValueError('Frozen encoder description changed')
    config = dict(load_config(ROOT/'configs/local.yaml'), seed=plan['seed'], embedding_dim=128,
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
        for source in [PLAN, ROOT/'docs/block_holdout_pilot_plan_v2_20261105.md', Path(__file__), ROOT/'src/eeg_decoding/data/pipeline_view.py',
                       ROOT/'src/eeg_decoding/evaluation/block_holdout.py', ROOT/'tests/test_block_holdout.py']:
            shutil.copyfile(source, output/'implementation'/source.name)
    _, decision, tests = run_pilot(records, text_path, config, plan, output, resume=True, status=lambda m: print(m, flush=True))
    before = json.loads(before_path.read_text(encoding='utf-8'))
    after = {p: sha(p) for p in before}
    if after != before:
        raise RuntimeError('A protected file changed')
    write_json(output/'preservation_after.json', {'unchanged': True, 'protected_files': len(before)})
    print(json.dumps(decision, indent=2), flush=True)
    print(json.dumps({'test_scored': sorted(tests)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/block_holdout_pilot_v2_20261105')
    parser.add_argument('--resume', action='store_true')
    arguments = parser.parse_args()
    main(arguments.output, arguments.resume)
