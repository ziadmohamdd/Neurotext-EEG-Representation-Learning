"""Run averaged-EEG test v5. v1 test sentences and the v2 test zone are never fitted or scored."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from eeg_decoding.data.dataset import observation_seed, read_manifest
from eeg_decoding.data.pipeline_view import center_crop, clean_signal
from eeg_decoding.evaluation.averaged_eeg import Layout, fold_audit, run_test
from eeg_decoding.evaluation.controls import gaussian_input
from eeg_decoding.evaluation.proximity import excluded_positions, forbidden_positions
from eeg_decoding.evaluation.sentence_holdout import sentence_holdout_split
from eeg_decoding.models.classical import eeg_features
from eeg_decoding.training.alignment_train import TextCache
from eeg_decoding.utils.logging import write_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'configs/averaged_eeg_v5_20261106.json'
APPROVED_SHA256 = 'd006d67e421de6f01191c159856bc1a182be8a15e7696920dac5d575aac68bf6'
WINDOW_SECONDS = 30
BINS = 20


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def time_bins(x):
    return np.concatenate([part.mean(-1) for part in np.array_split(x, BINS, axis=-1)]).astype(np.float32)


def build_features(rows, pipeline, seed, status=print):
    """One pass per observation. No reader normalization here; each fold scales from its own training rows."""
    arms = {name: [] for name in ('real', 'gaussian', 'binned_real', 'binned_gaussian')}
    durations = np.zeros(len(rows))
    for i, r in enumerate(rows):
        full = clean_signal(r, pipeline)
        durations[i] = full.shape[1] / r.sfreq
        x = center_crop(full, pipeline['window_samples'])
        noise = gaussian_input(x, np.random.default_rng(observation_seed(r, seed)))
        arms['real'].append(eeg_features(x, r.sfreq))
        arms['gaussian'].append(eeg_features(noise, r.sfreq))
        arms['binned_real'].append(time_bins(x))
        arms['binned_gaussian'].append(time_bins(noise))
        if i % 500 == 0:
            status(f'FEATURES {i} of {len(rows)}')
    return {name: np.stack(values) for name, values in arms.items()}, durations


def layout_from(records, plan):
    seed, spec = plan['seed'], plan['protected']['v2_test_zone']
    zone = excluded_positions(records, seed, **spec)
    v1_test = {(r.dataset, int(r.sentence_id)) for r in sentence_holdout_split(records, seed)['test']}
    folds = plan['folds']
    layout = Layout(records, zone, v1_test, forbidden_positions(records), spec['block'], folds['buffer_sentences'], folds['minimum_scored_per_block'])
    closed = {(release, i) for release, ids in zone.items() for i in ids}
    return layout, closed, v1_test


def check_sizes(layout, plan):
    expected = plan['folds']
    found = {release: sum(f.release == release for f in layout.folds) for release in layout.releases}
    if found != expected['blocks']:
        raise ValueError(f'Blocks differ from the frozen plan: {found}')
    for release in layout.releases:
        scored = {int(s) for f in layout.folds if f.release == release for s in f.sentences}
        rows = int(np.isin(layout.sentence, list(scored)).sum())
        if len(scored) != expected['scored'][f'{release}_sentences'] or rows != expected['scored'][f'{release}_observations']:
            raise ValueError(f'Scored sizes differ from the frozen plan in {release}: {len(scored)} sentences, {rows} observations')
    if len(layout.rows) != plan['cohort']['observations_usable'] or len(layout.identities) != plan['cohort']['sentences_usable']:
        raise ValueError('Usable sizes differ from the frozen plan')


def main(output, audit_only, resume):
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    manifests = [ROOT/p for p in plan['cohort']['manifests']]
    text_path = ROOT/plan['text_cache']
    records = [r for path in manifests for r in read_manifest(path)]
    if len(records) != plan['cohort']['observations_in_manifests'] or len({r.subject_key for r in records}) != plan['cohort']['readers']:
        raise ValueError('Cohort differs from the frozen plan')
    layout, closed, v1_test = layout_from(records, plan)
    check_sizes(layout, plan)
    cache = TextCache(text_path)
    audit = fold_audit(records, layout, cache, closed, v1_test, **plan['audit']['limits'])
    if audit_only:
        print(json.dumps(audit, indent=2))
        return
    if sha(PLAN) != APPROVED_SHA256:
        raise ValueError('Frozen plan changed; a change needs a new version')
    if output.exists() and not resume:
        raise FileExistsError('Use a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'leakage_audit.json', audit)
    if audit['failures']:
        raise ValueError('Audit failed: ' + '; '.join(audit['failures']))
    before_path = output/'preservation_before.json'
    if before_path.exists():
        before = json.loads(before_path.read_text(encoding='utf-8'))
    else:
        prior = sorted(p for p in (ROOT/'results').rglob('*') if p.is_file() and output not in p.parents)
        protected = prior + manifests + [text_path] + sorted({Path(r.signal_path) for r in records})
        print('Hashing protected files', flush=True)
        before = {str(p): sha(p) for p in protected}
        write_json(before_path, before)
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        write_json(output/'plan.json', dict(plan_sha256=APPROVED_SHA256, plan=plan, git_head=commit))
        (output/'implementation').mkdir()
        for source in [PLAN, ROOT/'docs/averaged_eeg_plan_v5_20261106.md', Path(__file__), ROOT/'src/eeg_decoding/evaluation/averaged_eeg.py', ROOT/'tests/test_averaged_eeg.py']:
            shutil.copyfile(source, output/'implementation'/source.name)
    stored = output/'features.npz'
    if stored.exists():
        with np.load(stored, allow_pickle=False) as f:
            features, durations = {name: f[name] for name in ('real', 'gaussian', 'binned_real', 'binned_gaussian')}, f['durations']
    else:
        spec = plan['inputs']['primary']
        pipeline = dict(notch_hz=spec['notch_hz'], drop_channels=spec['drop_channels'], window_samples=int(WINDOW_SECONDS * records[0].sfreq))
        features, durations = build_features(layout.rows, pipeline, plan['seed'], status=lambda m: print(m, flush=True))
        np.savez(stored, durations=durations, **features)
    text = cache.get(layout.keys).astype(np.float64)
    result = run_test(layout, text, features, durations, plan['arms']['position']['fourier_terms'], plan, status=lambda m: print(m, flush=True))
    write_json(output/'decision.json', result['decision'])
    write_json(output/'secondary.json', result['secondary'])
    write_json(output/'selection.json', result['selection'])
    np.savez_compressed(output/'sentence_scores.npz', **result['values'])
    if {p: sha(p) for p in before} != before:
        raise RuntimeError('A protected file changed')
    write_json(output/'preservation_after.json', {'unchanged': True, 'protected_files': len(before)})
    print(json.dumps(result['decision'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/averaged_eeg_v5_20261106')
    parser.add_argument('--audit-only', action='store_true', help='Check folds from manifests and text only; reads no EEG')
    parser.add_argument('--resume', action='store_true', help='Reuse hashes and features after a crash; scores nothing twice')
    arguments = parser.parse_args()
    main(arguments.output, arguments.audit_only, arguments.resume)
