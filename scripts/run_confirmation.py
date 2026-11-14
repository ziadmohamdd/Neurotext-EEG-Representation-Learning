"""Run confirmation v10: score the reserved v2 test blocks once with the frozen v5, v6 and v9 pipelines.

Buffers around the test blocks and v1 test sentences are never fitted or scored. `--audit-only` reads manifests and text
only, and no EEG. A full run first rebuilds every input and checks that the rows outside the test zone match the stored
v5 features and v6 word rows exactly, so no test score is computed if the inputs differ.
"""
import argparse
from dataclasses import fields
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from eeg_decoding.data.dataset import read_manifest
from eeg_decoding.evaluation import averaged_eeg, first_fixation, word_locked
from eeg_decoding.evaluation.averaged_eeg import fold_audit
from eeg_decoding.evaluation.confirmation import ConfirmLayout, ConfirmWordData, confirm_audit, read_claims, reserved_blocks
from eeg_decoding.evaluation.proximity import excluded_positions, forbidden_positions
from eeg_decoding.evaluation.sentence_holdout import sentence_holdout_split
from eeg_decoding.evaluation.word_locked import WordData
from eeg_decoding.training.alignment_train import TextCache
from eeg_decoding.utils.logging import write_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'configs/confirmation_v10_20261114.json'
APPROVED_SHA256 = '885e285e153d0d53da8f8d6e82243dc7d9290d98c498a8eab18333546dc91f6f'
CACHE = ROOT/'data/processed/confirmation_v10_20261114'
DEVELOPMENT = {'v5': 'averaged_eeg_v5_20261106', 'v6': 'word_locked_v6_20261106', 'v9': 'first_fixation_v9_20261114'}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v5_runner = load('averaged_eeg_v5', 'scripts/run_averaged_eeg.py')
v6_runner = load('word_locked_v6', 'scripts/run_word_locked.py')
sha = v6_runner.sha


def pipeline_plans(plan):
    plans = {}
    for key, name in (('v5', 'whole_sentence_v5'), ('v6', 'word_locked_v6'), ('v9', 'first_fixation_v9')):
        spec = plan['pipelines'][name]
        if sha(ROOT/spec['config']) != spec['sha256']:
            raise ValueError(f'A reused frozen plan changed: {spec["config"]}')
        plans[key] = json.loads((ROOT/spec['config']).read_text(encoding='utf-8'))
    if not plans['v5']['cohort']['manifests'] == plans['v6']['cohort']['manifests'] == plans['v9']['cohort']['manifests']:
        raise ValueError('The reused plans name different manifests')
    if plans['v5']['text_cache'] != plans['v6']['text_cache']:
        raise ValueError('The reused plans name different text caches')
    return plans


def build_layout(plan, plans):
    v6 = plans['v6']
    records = [r for path in v6['cohort']['manifests'] for r in read_manifest(ROOT/path)]
    seed, spec = plan['seed'], v6['protected']['v2_test_zone']
    blocks, zone = reserved_blocks(records, seed, **spec)
    if zone != excluded_positions(records, seed, **spec):
        raise ValueError('The test zone differs from the zone every earlier test kept closed')
    v1_test = {(r.dataset, int(r.sentence_id)) for r in sentence_holdout_split(records, seed)['test']}
    layout = ConfirmLayout(records, blocks, zone, v1_test, forbidden_positions(records), spec['block'], spec['buffer'], v6['folds']['minimum_scored_per_block'])
    tested = {(release, i) for release, runs in blocks.items() for run in runs for i in run}
    buffers = {(release, i) for release, ids in zone.items() for i in ids} - tested
    return records, layout, v1_test, buffers


def sizes_of(layout):
    scored = np.isin(layout.sentence, layout.scored)
    found = {'blocks': {release: sum(f.release == release for f in layout.folds) for release in layout.releases},
             'scored_sentences': int(len(layout.scored)), 'scored_observations': int(scored.sum()),
             'scored_readers': int(len(np.unique(layout.reader[scored]))), 'usable_observations': len(layout.rows), 'usable_sentences': len(layout.identities),
             'training_sentences': int(len(layout.identities) - len(layout.test_sentences))}
    for release in layout.releases:
        found[f'{release}_scored_sentences'] = int(sum(len(f.sentences) for f in layout.folds if f.release == release))
    return found


def audit_of(records, layout, v1_test, buffers, plan, plans, cache):
    audit = fold_audit(records, layout, cache, buffers, v1_test, **plans['v6']['audit']['limits'])
    audit['failures'] += confirm_audit(layout, plan['seed'])
    found = sizes_of(layout)
    expected = plan['sizes']
    if {k: found[k] for k in expected} != expected:
        audit['failures'].append(f'Sizes differ from the frozen plan: {found}')
    audit['sizes'] = found
    return audit


def sentence_features(layout, records, plans, output):
    """Rebuild v5 features for every usable row, and check the rows outside the test zone against stored v5 features."""
    stored = output/'features.npz'
    if stored.exists():
        with np.load(stored, allow_pickle=False) as f:
            return {name: f[name] for name in ('real', 'gaussian', 'binned_real', 'binned_gaussian')}, f['durations']
    v5 = plans['v5']
    spec = v5['inputs']['primary']
    pipeline = dict(notch_hz=spec['notch_hz'], drop_channels=spec['drop_channels'], window_samples=int(v5_runner.WINDOW_SECONDS * records[0].sfreq))
    features, durations = v5_runner.build_features(layout.rows, pipeline, v5['seed'], status=lambda m: print(m, flush=True))
    old_layout, _, _ = v5_runner.layout_from(records, v5)
    identity = lambda r: (r.dataset, r.subject_key, int(r.sentence_id))
    where = {identity(r): i for i, r in enumerate(layout.rows)}
    if len(where) != len(layout.rows):
        raise ValueError('Two usable rows share a reader and sentence')
    mine = np.array([where[identity(r)] for r in old_layout.rows])
    with np.load(ROOT/'results'/DEVELOPMENT['v5']/'features.npz', allow_pickle=False) as f:
        for name in ('real', 'gaussian', 'binned_real', 'binned_gaussian'):
            if not np.array_equal(f[name], features[name][mine]):
                raise ValueError(f'Rebuilt v5 features differ from the stored ones: {name}')
        if not np.array_equal(f['durations'], durations[mine]):
            raise ValueError('Rebuilt v5 durations differ from the stored ones')
    np.savez(stored, durations=durations, **features)
    return features, durations


def word_inputs(layout, revision, workers):
    """Extract word rows for every usable row into a new cache; rows outside the test zone must equal the stored v6 rows."""
    old_cache = v6_runner.CACHE
    v6_runner.CACHE = CACHE
    try:
        v6_runner.extract(layout, workers)
        tested = {layout.identities[s] for s in layout.test_sentences}
        for path in sorted((old_cache/'rows').glob('*.npz')):
            release = path.stem.split('_')[0]
            with np.load(path, allow_pickle=False) as old, np.load(CACHE/'rows'/path.name, allow_pickle=False) as new:
                keep = ~np.isin(new['sentence'], [i for r, i in tested if r == release])
                for name in old.files:
                    if not np.array_equal(old[name], new[name][keep], equal_nan=old[name].dtype.kind == 'f'):
                        raise ValueError(f'Rebuilt word rows differ from the stored v6 rows: {path.name} {name}')
        types = sorted({v6_runner.normal_token(t) for r in layout.rows for t in r.text.split()})
        target = CACHE/'word_embeddings.npz'
        if not target.exists():
            with np.load(old_cache/'word_embeddings.npz', allow_pickle=False) as f:
                known = dict(zip(f['types'].tolist(), f['vectors']))
            new = [t for t in types if t not in known]
            os.environ['HF_HUB_OFFLINE'] = '1'
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', revision=revision, device='cpu')
            encoded = dict(zip(new, model.encode(new, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)))
            vectors = np.stack([known[t] if t in known else encoded[t] for t in types]).astype(np.float32)
            np.savez_compressed(target, types=np.array(types), vectors=vectors,
                                metadata=json.dumps({'revision': revision, 'rule': 'lowercase, edge punctuation removed', 'reused_v6_types': len(types) - len(new),
                                                     'new_types': len(new)}))
        data, bands, waves, type_count = v6_runner.word_data(layout, revision)
    finally:
        v6_runner.CACHE = old_cache
    data = ConfirmWordData(**{f.name: getattr(data, f.name) for f in fields(WordData)})
    return data, bands, waves, type_count


def claims_from(results, plan):
    rule = plan['claims']
    minimum, margin = rule['minimum_gap_points'] / 100, rule['c3_margin_points'] / 100

    def item(decision, name):
        bounds = [decision['intervals'][name]['sentence_interval']]
        if name in decision.get('reader_intervals', {}):
            bounds.append(decision['reader_intervals'][name])
        return {'gap': decision['gaps'][name], 'bounds': [tuple(b) for b in bounds]}

    c1 = {key: item(results[key]['decision'], 'gaussian') for key in ('v5', 'v6', 'v9')}
    c2 = item(results['v6']['decision'], 'gaze_word')
    c3 = item(results['v9']['decision'], 'adds_to_gaze')
    return {'labels': read_claims(c1, c2, c3, minimum, margin), 'inputs': {'C1': c1, 'C2_real_minus_gaze': c2, 'C3': c3}}


def main(output, audit_only, workers, resume):
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    plans = pipeline_plans(plan)
    records, layout, v1_test, buffers = build_layout(plan, plans)
    cache = TextCache(ROOT/plans['v6']['text_cache'])
    audit = audit_of(records, layout, v1_test, buffers, plan, plans, cache)
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
    manifests = [ROOT/p for p in plans['v6']['cohort']['manifests']]
    before_path = output/'preservation_before.json'
    if before_path.exists():
        before = json.loads(before_path.read_text(encoding='utf-8'))
    else:
        prior = sorted(p for p in (ROOT/'results').rglob('*') if p.is_file() and output not in p.parents)
        stored = sorted((v6_runner.CACHE/'rows').glob('*.npz')) + [v6_runner.CACHE/'word_embeddings.npz']
        protected = prior + manifests + [ROOT/plans['v6']['text_cache']] + stored + sorted({Path(r.signal_path) for r in records})
        print('Hashing protected files', flush=True)
        before = {str(p): sha(p) for p in protected}
        write_json(before_path, before)
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        write_json(output/'plan.json', dict(plan_sha256=APPROVED_SHA256, plan=plan, git_head=commit))
        (output/'implementation').mkdir(exist_ok=True)
        for source in [PLAN, ROOT/'docs/confirmation_plan_v10_20261114.md', Path(__file__), ROOT/'src/eeg_decoding/evaluation/confirmation.py',
                       ROOT/'tests/test_confirmation.py']:
            shutil.copyfile(source, output/'implementation'/source.name)
    status = lambda m: print(m, flush=True)
    revision = cache.metadata['resolved_revision']
    status('INPUTS whole-sentence features')
    features, durations = sentence_features(layout, records, plans, output)
    status('INPUTS word rows')
    data, bands, waves, type_count = word_inputs(layout, revision, workers)
    counts = v6_runner.counts_of(data, type_count)
    write_json(output/'word_counts.json', counts)
    channels = [c for c in records[0].channels if c != 'Cz']
    if any([c for c in r.channels if c != 'Cz'] != channels for r in layout.rows) or waves.shape[1] != 6 * len(channels):
        raise ValueError('Channel lists or waveform width differ')
    anterior = np.array([channels.index(c) for c in plans['v9']['inputs']['anterior_channels']])
    text = cache.get(layout.keys).astype(np.float64)
    results = {}
    status('RUN whole-sentence pipeline (v5)')
    results['v5'] = averaged_eeg.run_test(layout, text, features, durations, plans['v5']['arms']['position']['fourier_terms'], plans['v5'], status=status)
    status('RUN word-locked pipeline (v6)')
    results['v6'] = word_locked.run_test(data, bands, waves, text, plans['v6']['arms']['position']['fourier_terms'], plans['v6'], status=status)
    status('RUN first-fixation pipeline (v9)')
    results['v9'] = first_fixation.run_test(data, waves, len(channels), anterior, plans['v9'], status=status)
    for key, result in results.items():
        folder = output/key
        folder.mkdir(exist_ok=True)
        for part in ('decision', 'secondary', 'selection'):
            write_json(folder/f'{part}.json', result[part])
        np.savez_compressed(folder/'sentence_scores.npz', **result['values'])
    claims = claims_from(results, plan)
    development = {key: json.loads((ROOT/'docs/research_results'/folder/'decision.json').read_text(encoding='utf-8'))['primary'] for key, folder in DEVELOPMENT.items()}
    claims['test_primary'] = {key: result['decision']['primary'] for key, result in results.items()}
    claims['development_primary'] = development
    write_json(output/'claims.json', claims)
    if {p: sha(p) for p in before} != before:
        raise RuntimeError('A protected file changed')
    write_json(output/'preservation_after.json', {'unchanged': True, 'protected_files': len(before)})
    print(json.dumps(claims['labels'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/confirmation_v10_20261114')
    parser.add_argument('--audit-only', action='store_true', help='Check the layout from manifests and text only; reads no EEG')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true', help='Reuse hashes and rebuilt inputs after a crash; scores nothing twice')
    arguments = parser.parse_args()
    main(arguments.output, arguments.audit_only, arguments.workers, arguments.resume)
