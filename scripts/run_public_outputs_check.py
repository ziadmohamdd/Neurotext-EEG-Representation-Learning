"""Run public outputs check v8 on two published models' generated text. No EEG file is opened and v1 test sentences and the v2 test zone stay closed."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

os.environ['HF_HUB_OFFLINE'] = '1'
import numpy as np
from eeg_decoding.data.dataset import read_manifest
from eeg_decoding.evaluation.position_bridge import bridge_scope
from eeg_decoding.evaluation.proximity import excluded_positions
from eeg_decoding.evaluation.public_outputs import label_corpus, read_outputs, run_check, squash
from eeg_decoding.evaluation.retrieval import normalize
from eeg_decoding.utils.logging import write_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'configs/public_outputs_check_v8_20261112.json'
APPROVED_SHA256 = '7b3f2ebdf646d9b812c78ab8aecf265f9ccee88e5ebf0115b8666344f1daf0f9'
MOVIE_REVIEWS = ('ZuCo1', 'task1')
csv.field_size_limit(10**8)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def glim_labels(folder):
    labels = {}
    for path in sorted(folder.glob('*.csv')):
        with open(path, encoding='utf-8', newline='') as f:
            for row in csv.DictReader(f):
                labels[squash(row['Input Text'])] = 'movie_review' if (row['Dataset'], row['Raw Task Key']) == MOVIE_REVIEWS else 'biography'
    return labels


def load_model(plan, source, folder, texts, protected, known):
    """Rows and labels for one model, with protected sentences removed. Returns the audit alongside."""
    arms, failures = plan['arms'][source['model']], []
    for name, facts in source['files'].items():
        if (folder/name).stat().st_size != facts['bytes'] or sha(folder/name) != facts['sha256']:
            failures.append(f'File differs from the plan: {name}')
    tables = {name: read_outputs(folder/name) for name in source['files']}
    base = tables[arms['real'][0]]
    targets = [r['target'] for r in base]
    for name, table in tables.items():
        if 'generation_results-7_22' not in name and [r['target'] for r in table] != targets:
            failures.append(f'Targets differ: {name}')
    sentences = sorted({squash(t): t for t in targets}.values())
    labelled = label_corpus(sentences, texts)
    closed = label_corpus(sentences, protected) == 'biography'
    disagree = [s for s, label in zip(sentences, labelled) if known.get(squash(s), label) != label]
    if disagree:
        failures.append(f'{len(disagree)} corpus labels disagree with GLIM')
    kept = [s for s, c in zip(sentences, closed) if not c]
    corpus = labelled[~closed]
    counts = plan['counts']
    found = {'rows_per_file': len(base), 'sentences_per_model': len(sentences), 'labelled': dict(Counter(labelled.tolist())),
             'protected_removed': int(closed.sum()), 'scored': dict(Counter(corpus.tolist()))}
    if found != counts:
        failures.append(f'Counts differ from the plan: {found}')
    if any(squash(s) in {squash(t) for t in protected} for s in kept):
        failures.append('A scored target matches a protected sentence')
    checked = sum(squash(s) in known for s in sentences)
    return tables, kept, corpus, dict(found, failures=failures, sentences_checked_against_glim=checked)


def arm_texts(table, field, place):
    rows = [r for r in table if squash(r['target']) in place]
    return [r[field] for r in rows], np.array([place[squash(r['target'])] for r in rows])


def main(output):
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    if sha(PLAN) != APPROVED_SHA256:
        raise ValueError('Frozen plan changed; a change needs a new version')
    if output.exists():
        raise FileExistsError('Use a fresh output directory')
    manifests = [ROOT/p for p in plan['cohort']['manifests']]
    records = [r for path in manifests for r in read_manifest(path)]
    zone = excluded_positions(records, plan['seed'], **plan['protected']['v2_test_zone'])
    _, _, closed, v1_test = bridge_scope(records, plan['seed'], zone)
    protected = sorted({r.text for r in records if (r.dataset, int(r.sentence_id)) in closed | v1_test})
    texts = sorted({r.text for r in records})
    known = glim_labels(ROOT/plan['corpus_labels']['glim_folder'])
    loaded, audit = {}, {}
    for source in plan['files']['sources']:
        model = source['model']
        *loaded[model], audit[model] = load_model(plan, source, ROOT/plan['files']['folder']/model, texts, protected, known)
    output.mkdir(parents=True)
    write_json(output/'leakage_audit.json', audit)
    failures = [f'{model}: {f}' for model, a in audit.items() for f in a['failures']]
    if failures:
        raise ValueError('Audit failed: ' + '; '.join(failures))
    prior = sorted(p for p in (ROOT/'results').rglob('*') if p.is_file() and output not in p.parents)
    before = {str(p): sha(p) for p in prior + manifests}
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    write_json(output/'plan.json', dict(plan_sha256=APPROVED_SHA256, plan=plan, git_head=commit))
    (output/'implementation').mkdir()
    for source in [PLAN, ROOT/'docs/public_outputs_check_plan_v8_20261112.md', Path(__file__), ROOT/'src/eeg_decoding/evaluation/public_outputs.py', ROOT/'tests/test_public_outputs.py']:
        shutil.copyfile(source, output/'implementation'/source.name)
    from sentence_transformers import SentenceTransformer
    embedding = plan['inputs']['embedding']
    encoder = SentenceTransformer(embedding['model'], revision=embedding['revision'], device='cpu')

    def encode(sentences):
        return normalize(encoder.encode(sentences, batch_size=64, show_progress_bar=False, convert_to_numpy=True).astype(np.float64))

    decisions, secondaries = {}, {}
    for model, (tables, kept, corpus) in loaded.items():
        arms = plan['arms'][model]
        place = {squash(s): i for i, s in enumerate(kept)}
        bank = encode(kept)
        names = {'real': arms['real'], 'noise': arms['noise'], **arms['secondary']}
        vectors, own, variety = {}, {}, {}
        for name, (file, field) in names.items():
            generated, own[name] = arm_texts(tables[file], field, place)
            vectors[name] = encode(generated)
            top = Counter(generated).most_common(1)[0][1]
            variety[name] = {'rows': len(generated), 'distinct_outputs': len(set(generated)), 'most_common_share': top / len(generated)}
        result = run_check({k: vectors[k] for k in ('real', 'noise')}, own['real'], bank, corpus, plan)
        extra = {}
        for name in arms['secondary']:
            # Secondary arms are scored the same way against the same noise arm; their labels are never used.
            if not np.array_equal(own[name], own['real']):
                extra[name] = {'skipped': 'Its rows do not line up with the plain file'}
                continue
            scored = run_check({'real': vectors[name], 'noise': vectors['noise']}, own[name], bank, corpus, plan)
            extra[name] = {'scores': scored['decision']['scores']['real'], 'n_way': scored['secondary']['n_way']['real'], 'two_way_split': scored['secondary']['two_way_split']['real']}
        decisions[model] = result['decision']
        secondaries[model] = dict(result['secondary'], other_arms=extra, variety=variety)
        np.savez_compressed(output/f'sentence_scores_{model}.npz', **result['values'])
        print(model, result['decision']['label'], flush=True)
    write_json(output/'decision.json', decisions)
    write_json(output/'secondary.json', secondaries)
    if {p: sha(p) for p in before} != before:
        raise RuntimeError('A protected file changed')
    write_json(output/'preservation_after.json', {'unchanged': True, 'protected_files': len(before)})
    print(json.dumps(decisions, indent=2), flush=True)
    print(json.dumps(secondaries, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/public_outputs_check_v8_20261112')
    main(parser.parse_args().output)
