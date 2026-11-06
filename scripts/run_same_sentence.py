"""Run same-sentence test v7 from stored v5 features. No EEG file is opened and v1 test sentences and the v2 test zone stay closed."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from eeg_decoding.data.dataset import read_manifest
from eeg_decoding.evaluation.averaged_eeg import Layout
from eeg_decoding.evaluation.proximity import excluded_positions, forbidden_positions
from eeg_decoding.evaluation.same_sentence import Pairing, run_test, shared_pairs
from eeg_decoding.evaluation.sentence_holdout import sentence_holdout_split
from eeg_decoding.utils.logging import write_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'configs/same_sentence_v7_20261106.json'
APPROVED_SHA256 = '5810ce7d4bf35adac3b3a5052c146c44460c357a18ae5670a44851dc52f38471'
ARMS = ('real', 'gaussian', 'binned_real', 'binned_gaussian')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def audit(records, layout, pairing, protected, plan, stored):
    failures = []
    releases = pairing.releases
    pairs = plan['pairs']
    if len(pairing.pairs) != pairs['sentences']:
        failures.append(f'Pairs differ from the plan: {len(pairing.pairs)}')
    if [int((pairing.side == side).sum()) for side in (0, 1)] != pairs['observations']:
        failures.append('Observations differ from the plan')
    if any((release, ids[side]) in protected for ids in pairing.pairs for side, release in enumerate(releases)):
        failures.append('A pair includes a protected sentence')
    usable = {(r.dataset, int(r.sentence_id)) for r in layout.rows}
    if usable & protected:
        failures.append('A protected sentence is in the usable rows')
    if len({tuple(r.channels) for r in records}) != 1:
        failures.append('Channel lists differ between rows')
    if len(layout.rows) != plan['cohort']['observations_usable']:
        failures.append('Usable rows differ from the plan')
    if any(len(stored[name]) != len(layout.rows) for name in stored):
        failures.append('Stored features do not match the usable rows')
    return {'failures': failures, 'pairs': len(pairing.pairs), 'observations': [int((pairing.side == side).sum()) for side in (0, 1)],
            'usable_observations': len(layout.rows), 'protected_sentences': len(protected)}


def main(output):
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    if sha(PLAN) != APPROVED_SHA256:
        raise ValueError('Frozen plan changed; a change needs a new version')
    if output.exists():
        raise FileExistsError('Use a fresh output directory')
    features_path = ROOT/plan['inputs']['features_file']
    if sha(features_path) != plan['inputs']['features_sha256']:
        raise ValueError('Stored features changed')
    manifests = [ROOT/p for p in plan['cohort']['manifests']]
    records = [r for path in manifests for r in read_manifest(path)]
    seed, spec = plan['seed'], plan['protected']['v2_test_zone']
    zone = excluded_positions(records, seed, **spec)
    v1_test = {(r.dataset, int(r.sentence_id)) for r in sentence_holdout_split(records, seed)['test']}
    protected = {(release, i) for release, ids in zone.items() for i in ids} | v1_test
    layout = Layout(records, zone, v1_test, forbidden_positions(records), spec['block'], 10, 10)
    releases, pairs = shared_pairs(records, protected)
    pairing = Pairing(layout.rows, pairs, releases, layout.place)
    with np.load(features_path, allow_pickle=False) as f:
        features, durations = {name: f[name] for name in ARMS}, f['durations']
    checked = audit(records, layout, pairing, protected, plan, dict(features, durations=durations))
    output.mkdir(parents=True)
    write_json(output/'leakage_audit.json', checked)
    if checked['failures']:
        raise ValueError('Audit failed: ' + '; '.join(checked['failures']))
    prior = sorted(p for p in (ROOT/'results').rglob('*') if p.is_file() and output not in p.parents)
    before = {str(p): sha(p) for p in prior + manifests}
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    write_json(output/'plan.json', dict(plan_sha256=APPROVED_SHA256, plan=plan, git_head=commit))
    (output/'implementation').mkdir()
    for source in [PLAN, ROOT/'docs/same_sentence_plan_v7_20261106.md', Path(__file__), ROOT/'src/eeg_decoding/evaluation/same_sentence.py', ROOT/'tests/test_same_sentence.py']:
        shutil.copyfile(source, output/'implementation'/source.name)
    result = run_test(pairing, features, durations, plan, status=lambda m: print(m, flush=True))
    write_json(output/'decision.json', result['decision'])
    write_json(output/'secondary.json', result['secondary'])
    np.savez_compressed(output/'sentence_scores.npz', **result['values'])
    if {p: sha(p) for p in before} != before:
        raise RuntimeError('A protected file changed')
    write_json(output/'preservation_after.json', {'unchanged': True, 'protected_files': len(before)})
    print(json.dumps(result['decision'], indent=2), flush=True)
    print(json.dumps(result['secondary'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/same_sentence_v7_20261106')
    main(parser.parse_args().output)
