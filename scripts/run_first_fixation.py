"""Run first-fixation test v9 from the stored v6 word rows. No EEG file is opened and v1 test sentences and the v2 test zone are never fitted or scored."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from eeg_decoding.data.dataset import read_manifest
from eeg_decoding.evaluation.averaged_eeg import fold_audit
from eeg_decoding.evaluation.first_fixation import run_test
from eeg_decoding.training.alignment_train import TextCache
from eeg_decoding.utils.logging import write_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'configs/first_fixation_v9_20261114.json'
APPROVED_SHA256 = 'ce55bbfa603ca82019920b0622ace6aa8feac1dfbbcb06ce8a9282b4703c5b79'
spec = importlib.util.spec_from_file_location('word_locked_v6', ROOT/'scripts/run_word_locked.py')
v6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v6)
sha = v6.sha


def rows_digest(folder):
    digest = hashlib.sha256()
    for path in sorted(folder.glob('*.npz')):
        digest.update(path.name.encode())
        digest.update(bytes.fromhex(sha(path)))
    return digest.hexdigest()


def main(output):
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    if sha(PLAN) != APPROVED_SHA256:
        raise ValueError('Frozen plan changed; a change needs a new version')
    if output.exists():
        raise FileExistsError('Use a fresh output directory')
    cache = plan['inputs']['cache']
    if rows_digest(ROOT/cache['rows_folder']) != cache['rows_digest'] or sha(v6.CACHE/'word_embeddings.npz') != cache['word_embeddings_sha256']:
        raise ValueError('Stored word rows changed')
    manifests = [ROOT/p for p in plan['cohort']['manifests']]
    records = [r for path in manifests for r in read_manifest(path)]
    layout, closed, v1_test = v6.layout_from(records, plan)
    v6.check_sizes(layout, plan)
    text = TextCache(ROOT/plan['text_cache'])
    audit = fold_audit(records, layout, text, closed, v1_test, **plan['audit']['limits'])
    data, _, waves, type_count = v6.word_data(layout, text.metadata['resolved_revision'])
    counts = v6.counts_of(data, type_count)
    if {k: counts[k] for k in plan['cohort']['word_rows']} != plan['cohort']['word_rows']:
        audit['failures'].append(f'Word rows differ from the plan: {counts}')
    stored = {(layout.identities[s][0], int(layout.identities[s][1])) for s in np.unique(data.sentence)}
    if stored & (closed | v1_test):
        audit['failures'].append('A protected sentence is in the stored word rows')
    channels = [c for c in records[0].channels if c != 'Cz']
    if any([c for c in r.channels if c != 'Cz'] != channels for r in layout.rows) or waves.shape[1] != 6 * len(channels):
        audit['failures'].append('Channel lists or waveform width differ')
    missing = sorted(set(plan['inputs']['anterior_channels']) - set(channels))
    if missing:
        audit['failures'].append(f'Anterior channels not found: {missing}')
    output.mkdir(parents=True)
    write_json(output/'leakage_audit.json', audit)
    write_json(output/'word_counts.json', counts)
    if audit['failures']:
        raise ValueError('Audit failed: ' + '; '.join(audit['failures']))
    prior = sorted(p for p in (ROOT/'results').rglob('*') if p.is_file() and output not in p.parents)
    protected = prior + manifests + [ROOT/plan['text_cache']] + sorted((ROOT/cache['rows_folder']).glob('*.npz'))
    before = {str(p): sha(p) for p in protected}
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    write_json(output/'plan.json', dict(plan_sha256=APPROVED_SHA256, plan=plan, git_head=commit))
    (output/'implementation').mkdir()
    for source in [PLAN, ROOT/'docs/first_fixation_plan_v9_20261114.md', Path(__file__), ROOT/'src/eeg_decoding/evaluation/first_fixation.py',
                   ROOT/'src/eeg_decoding/evaluation/word_locked.py', ROOT/'tests/test_first_fixation.py']:
        shutil.copyfile(source, output/'implementation'/source.name)
    anterior = np.array([channels.index(c) for c in plan['inputs']['anterior_channels']])
    result = run_test(data, waves, len(channels), anterior, plan, status=lambda m: print(m, flush=True))
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
    parser.add_argument('--output', type=Path, default=ROOT/'results/first_fixation_v9_20261114')
    main(parser.parse_args().output)
