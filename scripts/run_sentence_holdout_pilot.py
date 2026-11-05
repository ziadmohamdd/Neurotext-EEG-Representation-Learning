"""Run the frozen sentence-holdout pilot. Test rows are read only after a gate pass."""
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
from eeg_decoding.data.validation import manifest_digest
from eeg_decoding.evaluation.controls import shuffled_keys
from eeg_decoding.evaluation.sentence_holdout import (apply_gate, report_scores, score_queries,
                                                       sentence_groups, sentence_holdout_split, word_count)
from eeg_decoding.evaluation.spectral_pilot import count_basis
from eeg_decoding.models.classical import ridge_path
from eeg_decoding.models.eeg_cnn import EEGCNN
from eeg_decoding.models.projection import EEGTextModel
from eeg_decoding.preprocessing.normalization import ChannelNormalizer
from eeg_decoding.training.alignment_train import TextCache, train_alignment
from eeg_decoding.training.checkpoints import load_checkpoint
from eeg_decoding.utils.config import load_config
from eeg_decoding.utils.device import device_info
from eeg_decoding.utils.logging import write_json
from eeg_decoding.utils.reproducibility import seed_all
from eeg_decoding.workflow import feature_matrix, loader, prepare_metadata

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'configs/sentence_holdout_pilot_v1_20261105.json'
APPROVED_SHA256 = 'e66ecee76d49f40c42bbffd3ac92481440c81da7b421cc0989d4a7a1edb84bda'
PRIMARY = 10
CONTROLS = ('real', 'gaussian', 'shuffled')


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


def protected_hashes(paths):
    return {str(p): sha(p) for p in paths}


def ridge_predict(features, weights):
    return ((features - weights['mean']) / weights['scale']) @ weights['coef'].T + weights['intercept']


def select_ridge(train, targets, val, rows, text, alphas, ways, matched_size):
    """First grid entry wins ties, as the plan states."""
    best, chosen, table = -np.inf, None, []
    for alpha, weights in ridge_path(train, targets, alphas):
        scored = score_queries(ridge_predict(val, weights), rows, text, ways, PRIMARY, matched_size)
        metric = float(np.mean(list(report_scores(scored, rows, PRIMARY)['primary_per_reader'].values())))
        table.append({'alpha': alpha, 'validation_primary': metric})
        if metric > best:
            best, chosen = metric, dict(weights, alpha=alpha)
    return chosen, table


def timing_features(rows, levels_from=None):
    """Observation duration and word count only; no amplitudes enter this arm."""
    durations, counts = [], []
    for r in rows:
        with np.load(r.signal_path, allow_pickle=False) as f:
            durations.append(f['eeg'].shape[1] / r.sfreq)
        counts.append(word_count(r.text))
    counts = np.array(counts, dtype=float)
    basis, _ = count_basis(levels_from if levels_from is not None else counts, counts)
    return np.c_[durations, counts, basis]


def predict_cnn(model, rows, config, norm, control, device):
    model.to(device).eval()
    output = []
    with torch.no_grad():
        for batch in loader(rows, config, norm, control='gaussian' if control == 'gaussian' else 'real'):
            output.append(model(batch['eeg'].to(device), batch['mask'].to(device)).float().cpu().numpy())
    return np.concatenate(output)


def build_cnn(channels, config, text_dim):
    return EEGTextModel(EEGCNN(channels, config['embedding_dim'], quality_aware=False, temporal_segments=1),
                        config['embedding_dim'], text_dim)


def run_pilot(records, text_path, cnn_config, plan, output, ways=(2, 4, 10, 24), matched_size=20,
              readers_positive=20, iterations=10000, resume=False, status=print):
    """Validation arms, then the gate, then one test evaluation per passing model."""
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError('Use a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    seed = plan['seed']
    splits = sentence_holdout_split(records, seed, plan['split']['fractions']['test'])
    if {r.subject_key for r in splits['train']} != {r.subject_key for r in splits['val']}:
        raise ValueError('Validation readers must also appear in training')
    text = TextCache(text_path)
    train_rows, val_rows = splits['train'], splits['val']
    norm, meta = prepare_metadata(records, splits, cnn_config, 'alignment', 'sentence_holdout')
    device, info = device_info(cnn_config.get('device', 'auto'))
    meta.update(text_cache_digest=text.digest, encoder_origin='supervised_cnn', device_info=info)
    write_json(output/'splits.json', meta['splits'])
    alphas = plan['models']['ridge']['alphas']
    max_seconds = plan['models']['cnn']['max_hours_per_arm'] * 3600
    train_keys = [r.key for r in train_rows]
    fit = {'train': train_rows, 'val': val_rows}

    def finished(folder):
        path = output/folder/'validation.json'
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None

    def record(folder, rows, predicted, split, **extra):
        scored = score_queries(predicted, rows, text, ways, PRIMARY, matched_size)
        report = dict(report_scores(scored, rows, PRIMARY), evaluation_split=split,
                      manifest_digest=manifest_digest(rows), **extra)
        name = 'validation.json' if split == 'val' else 'test.json'
        if split == 'test' and (output/folder/name).exists():
            raise FileExistsError('Test sentences may be scored only once')
        write_json(output/folder/name, report)
        return report

    reports, models = {}, {}
    # Timing first: it needs no amplitudes and sets the structural comparator.
    if not (reports.setdefault('timing', finished('timing')) or None):
        status('RUN timing')
        (output/'timing').mkdir(exist_ok=True)
        counts = np.array([word_count(r.text) for r in train_rows], dtype=float)
        x = timing_features(train_rows, counts)
        v = timing_features(val_rows, counts)
        weights, table = select_ridge(x, text.get(train_keys).astype(np.float64), v, val_rows, text, alphas, ways, matched_size)
        np.savez_compressed(output/'timing/ridge.npz', **weights)
        reports['timing'] = record('timing', val_rows, ridge_predict(v, weights), 'val', selection=table,
                                   selected_alpha=float(weights['alpha']))
    for control in CONTROLS:
        folder = 'ridge_'+control
        reports[folder] = finished(folder)
        if reports[folder]:
            continue
        status('RUN '+folder)
        (output/folder).mkdir(exist_ok=True)
        input_control = 'gaussian' if control == 'gaussian' else 'real'
        matrices = {name: feature_matrix(rows, cnn_config, norm, output/folder/f'{name}_features.npy', control=input_control)
                    for name, rows in fit.items()}
        keys = shuffled_keys(train_keys, seed) if control == 'shuffled' else train_keys
        weights, table = select_ridge(matrices['train'], text.get(keys).astype(np.float64), matrices['val'],
                                      val_rows, text, alphas, ways, matched_size)
        np.savez_compressed(output/folder/'ridge.npz', **weights)
        reports[folder] = record(folder, val_rows, ridge_predict(matrices['val'], weights), 'val', selection=table,
                                 selected_alpha=float(weights['alpha']))
    for control in CONTROLS:
        folder = 'cnn_'+control
        reports[folder] = finished(folder)
        if reports[folder]:
            continue
        status('RUN '+folder)
        (output/folder).mkdir(exist_ok=True)
        seed_all(seed)
        model = build_cnn(len(meta['channels']), cnn_config, text.embeddings.shape[1])
        deadline = time.monotonic() + max_seconds
        train = TimedLoader(loader(train_rows, cnn_config, norm, True, True, control), deadline)
        val = TimedLoader(loader(val_rows, cnn_config, norm, False, True, 'gaussian' if control == 'gaussian' else 'real'), deadline)
        last = output/folder/'last.pt'
        started = time.monotonic()
        history = train_alignment(model, train, val, text, cnn_config, device, output/folder,
                                  dict(meta, config=cnn_config, training_control=control), last if last.exists() else None)
        best = load_checkpoint(output/folder/'best.pt', model)
        reports[folder] = record(folder, val_rows, predict_cnn(model, val_rows, cnn_config, norm, control, device), 'val',
                                 best_epoch=best['epoch'], epochs_run=len(history), validation_loss=best['best'],
                                 seconds=time.monotonic()-started)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    primary = {name: np.array([q['primary'] for q in r['queries']]) for name, r in reports.items()}
    decision = {}
    for family in ('ridge', 'cnn'):
        comparators = {'gaussian': primary[family+'_gaussian'], 'class_shuffled': primary[family+'_shuffled'],
                       'timing_only_ridge': primary['timing']}
        decision[family] = apply_gate(primary[family+'_real'], comparators, val_rows,
                                      readers_positive=readers_positive, iterations=iterations)
    write_json(output/'decision.json', decision)
    # The gate is settled above. Only now may test rows be read, and only for a pass.
    tests = {}
    for family, result in decision.items():
        if not result['passed']:
            continue
        status('TEST '+family)
        test_rows = splits['test']
        counts = np.array([word_count(r.text) for r in train_rows], dtype=float)
        if 'timing' not in tests:
            with np.load(output/'timing/ridge.npz', allow_pickle=False) as f:
                tests['timing'] = record('timing', test_rows, ridge_predict(timing_features(test_rows, counts), dict(f)), 'test')
        for control in CONTROLS:
            folder = f'{family}_{control}'
            if family == 'ridge':
                x = feature_matrix(test_rows, cnn_config, norm, output/folder/'test_features.npy',
                                   control='gaussian' if control == 'gaussian' else 'real')
                with np.load(output/folder/'ridge.npz', allow_pickle=False) as f:
                    predicted = ridge_predict(x, dict(f))
            else:
                model = build_cnn(len(meta['channels']), cnn_config, text.embeddings.shape[1])
                load_checkpoint(output/folder/'best.pt', model)
                predicted = predict_cnn(model, test_rows, cnn_config, norm, control, device)
            tests[folder] = record(folder, test_rows, predicted, 'test')
    return reports, decision, tests


def main(output, resume):
    if sha(PLAN) != APPROVED_SHA256:
        raise ValueError('Frozen plan changed; a change needs a new version')
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    manifests = [ROOT/p for p in plan['cohort']['manifests']]
    text_path = ROOT/plan['text_cache']
    records = [r for path in manifests for r in read_manifest(path)]
    readers = {r.subject_key for r in records}
    if len(records) != plan['cohort']['observations'] or len(readers) != plan['cohort']['readers']:
        raise ValueError('Cohort differs from the frozen plan')
    splits = sentence_holdout_split(records, plan['seed'], plan['split']['fractions']['test'])
    for name, rows in splits.items():
        expected = plan['split']['expected_sizes'][name]
        if len(rows) != expected['observations'] or len({r.key for r in rows}) != expected['sentences']:
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
                  patience=cnn['patience'], temporal_segments=1, max_samples=1600, quality_aware=False)
    protected = sorted(p for p in (ROOT/'results').rglob('*') if p.is_file() and output.resolve() not in p.resolve().parents)
    protected += manifests + [text_path] + sorted({Path(r.signal_path) for r in records})
    before_path = output/'preservation_before.json'
    output.mkdir(parents=True, exist_ok=resume)
    if not before_path.exists():
        print('Hashing protected files', flush=True)
        write_json(before_path, protected_hashes(protected))
        groups, merged = sentence_groups(records)
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        write_json(output/'plan.json', dict(plan_sha256=APPROVED_SHA256, plan=plan, config=config, git_head=commit,
                                            sentence_groups=len(groups), merged_pairs=merged))
        for source in [PLAN, ROOT/'docs/sentence_holdout_pilot_plan_v1_20261105.md', Path(__file__),
                       ROOT/'src/eeg_decoding/evaluation/sentence_holdout.py', ROOT/'tests/test_sentence_holdout.py']:
            (output/'implementation').mkdir(exist_ok=True)
            shutil.copyfile(source, output/'implementation'/source.name)
    reports, decision, tests = run_pilot(records, text_path, config, plan, output, resume=True,
                                         status=lambda m: print(m, flush=True))
    before = json.loads(before_path.read_text(encoding='utf-8'))
    after = protected_hashes([Path(p) for p in before])
    if after != before:
        raise RuntimeError('A protected file changed')
    write_json(output/'preservation_after.json', {'unchanged': True, 'protected_files': len(before)})
    print(json.dumps(decision, indent=2), flush=True)
    print(json.dumps({'test_scored': sorted(tests)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/sentence_holdout_pilot_v1_20261105')
    parser.add_argument('--resume', action='store_true')
    arguments = parser.parse_args()
    main(arguments.output, arguments.resume)
