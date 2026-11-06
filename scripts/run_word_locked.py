"""Run word-locked test v6. v1 test sentences and the v2 test zone are never stored, fitted or scored."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import string
import subprocess

import numpy as np
from eeg_decoding.data.dataset import read_manifest
from eeg_decoding.data.word_rows import extract_rows
from eeg_decoding.evaluation.averaged_eeg import Layout, fold_audit
from eeg_decoding.evaluation.proximity import excluded_positions, forbidden_positions
from eeg_decoding.evaluation.sentence_holdout import sentence_holdout_split
from eeg_decoding.evaluation.word_locked import WordData, run_test
from eeg_decoding.training.alignment_train import TextCache
from eeg_decoding.utils.logging import write_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'configs/word_locked_v6_20261106.json'
APPROVED_SHA256 = '3f4053515a3128dfc3269ad44e8eb6ca2cc6aa6cd4ad1a326112b323b7fca8ff'
CACHE = ROOT/'data/processed/word_locked_v6_20261106'
EDGE = string.punctuation + '“”‘’–—…'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def normal_token(token):
    lowered = token.lower()
    return lowered.strip(EDGE) or lowered


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
            raise ValueError(f'Scored sizes differ from the frozen plan in {release}')
    if len(layout.rows) != plan['cohort']['observations_usable'] or len(layout.identities) != plan['cohort']['sentences_usable']:
        raise ValueError('Usable sizes differ from the frozen plan')


def extract_one(source, digest, wanted, keep, target):
    """One reader's file. Usable sentences only; every other sentence's word field is skipped."""
    if sha(source) != digest:
        raise ValueError(f'Source hash differs: {source}')
    rows = extract_rows(source, wanted, keep)
    temporary = Path(str(target) + '.partial.npz')
    np.savez(temporary, **rows)
    temporary.replace(target)
    return target.name, len(rows['slot'])


def extract(layout, workers):
    (CACHE/'rows').mkdir(parents=True, exist_ok=True)
    jobs = {}
    for r in layout.rows:
        name = r.subject_key.replace(':', '_')
        if int(r.sentence_id) != r.provenance['matlab_sentence_index'] - 1:
            raise ValueError('Sentence ID differs from its source index')
        job = jobs.setdefault(name, dict(source=r.provenance['source'], digest=r.provenance['source_sha256'], wanted={},
                                         keep=np.array([i for i, c in enumerate(r.channels) if c != 'Cz'])))
        job['wanted'][int(r.sentence_id)] = len(r.text.split())
    pending = {n: j for n, j in jobs.items() if not (CACHE/'rows'/f'{n}.npz').exists()}
    print(f'Extracting {len(pending)} of {len(jobs)} readers', flush=True)
    with ProcessPoolExecutor(workers) as pool:
        futures = [pool.submit(extract_one, target=CACHE/'rows'/f'{n}.npz', **j) for n, j in pending.items()]
        for future in as_completed(futures):
            print(*future.result(), flush=True)


def build_embeddings(types, revision):
    path = CACHE/'word_embeddings.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as f:
            if f['types'].tolist() != types:
                raise ValueError('Word cache differs from this layout')
            return f['vectors'].astype(np.float64)
    os.environ['HF_HUB_OFFLINE'] = '1'
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', revision=revision, device='cpu')
    vectors = model.encode(types, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    np.savez_compressed(path, types=np.array(types), vectors=vectors, metadata=json.dumps({'revision': revision, 'rule': 'lowercase, edge punctuation removed'}))
    return vectors.astype(np.float64)


def word_data(layout, revision):
    token_lists = {}
    for r in layout.rows:
        token_lists[layout.lookup[r.dataset, int(r.sentence_id)]] = [normal_token(t) for t in r.text.split()]
    types = sorted({t for row in token_lists.values() for t in row})
    where = {t: i for i, t in enumerate(types)}
    tokens = [np.array([where[t] for t in token_lists[i]], dtype=np.int64) for i in range(len(layout.identities))]
    readers = {name: i for i, name in enumerate(layout.readers)}
    release_of = {r.subject_key: r.dataset for r in layout.rows}
    parts = {name: [] for name in ('reader', 'sentence', 'slot', 'durations', 'fixations', 'bands', 'waves')}
    for name, index in readers.items():
        with np.load(CACHE/'rows'/f"{name.replace(':', '_')}.npz", allow_pickle=False) as f:
            count = len(f['slot'])
            parts['reader'].append(np.full(count, index))
            parts['sentence'].append(np.array([layout.lookup[release_of[name], int(i)] for i in f['sentence']], dtype=np.int64))
            for key, field in (('slot', 'slot'), ('durations', 'durations'), ('fixations', 'fixations'), ('bands', 'bands'), ('waves', 'waves')):
                parts[key].append(f[field])
    joined = {name: np.concatenate(values) for name, values in parts.items()}
    vectors = build_embeddings(types, revision)
    data = WordData(layout, joined['reader'], joined['sentence'], joined['slot'], joined['durations'].astype(np.float64), joined['fixations'].astype(np.float64),
                    tokens, vectors, np.array([len(t) for t in types], dtype=np.float64))
    return data, joined['bands'], joined['waves'], len(types)


def counts_of(data, type_count):
    scored = np.isin(data.sentence, data.layout.scored)
    return {'word_rows': int(len(data.reader)), 'scored_word_rows': int(scored.sum()), 'word_types': type_count,
            'finite_duration_share': float(np.isfinite(data.durations).mean()), 'finite_fixation_share': float(np.isfinite(data.fixations).mean()),
            'readers_with_rows': int(len(np.unique(data.reader)))}


def main(output, mode, workers, resume):
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    manifests = [ROOT/p for p in plan['cohort']['manifests']]
    text_path = ROOT/plan['text_cache']
    records = [r for path in manifests for r in read_manifest(path)]
    layout, closed, v1_test = layout_from(records, plan)
    check_sizes(layout, plan)
    cache = TextCache(text_path)
    revision = cache.metadata['resolved_revision']
    audit = fold_audit(records, layout, cache, closed, v1_test, **plan['audit']['limits'])
    if audit['failures']:
        raise ValueError('Audit failed: ' + '; '.join(audit['failures']))
    if mode == 'extract':
        extract(layout, workers)
        data, bands, waves, type_count = word_data(layout, revision)
        print(json.dumps(dict(counts_of(data, type_count), bands=list(bands.shape), waves=list(waves.shape), revision=revision, audit=audit), indent=2), flush=True)
        return
    if sha(PLAN) != APPROVED_SHA256:
        raise ValueError('Frozen plan changed; a change needs a new version')
    if output.exists() and not resume:
        raise FileExistsError('Use a fresh output directory')
    data, bands, waves, type_count = word_data(layout, revision)
    counts = counts_of(data, type_count)
    frozen = plan['cohort']['word_rows']
    if {k: counts[k] for k in frozen} != frozen:
        raise ValueError(f'Word rows differ from the frozen plan: {counts}')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'leakage_audit.json', audit)
    write_json(output/'word_counts.json', counts)
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
        (output/'implementation').mkdir(exist_ok=True)
        for source in [PLAN, ROOT/'docs/word_locked_plan_v6_20261106.md', Path(__file__), ROOT/'src/eeg_decoding/evaluation/word_locked.py',
                       ROOT/'src/eeg_decoding/data/word_rows.py', ROOT/'tests/test_word_locked.py']:
            shutil.copyfile(source, output/'implementation'/source.name)
    sentence_text = cache.get(layout.keys).astype(np.float64)
    result = run_test(data, bands, waves, sentence_text, plan['arms']['position']['fourier_terms'], plan, status=lambda m: print(m, flush=True))
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
    parser.add_argument('--output', type=Path, default=ROOT/'results/word_locked_v6_20261106')
    parser.add_argument('--extract', action='store_true', help='Store usable word rows and print exact counts; fits and scores nothing')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true', help='Reuse hashes after a crash')
    arguments = parser.parse_args()
    main(arguments.output, 'extract' if arguments.extract else 'run', arguments.workers, arguments.resume)
