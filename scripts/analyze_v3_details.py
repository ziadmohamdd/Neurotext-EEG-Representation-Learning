"""Exploratory breakdown of saved v3 scores, added after the run. Nothing is trained and no decision changes."""
import json
from pathlib import Path
import numpy as np
from eeg_decoding.data.dataset import read_manifest
from eeg_decoding.evaluation.block_holdout import _macro, bootstrap_gap
from eeg_decoding.evaluation.proximity import excluded_positions, proximity_folds, rows_for
from eeg_decoding.utils.logging import write_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'results/proximity_attribution_v3_20261105'
OUTPUT = ROOT/'results/proximity_v3_details_20261105'
CNN_ARMS = ['F_full_near_real', 'F_full_far_real', 'F_full_near_gaussian', 'F_full_far_gaussian', 'F_full_near_shuffled', 'F_full_far_shuffled',
            'F_half_near_real', 'F_half_far_real', 'v1_full_near_real', 'v1_full_far_real']


def load(name):
    return [json.loads((SOURCE/f'fold{fold}'/name/'scored.json').read_text(encoding='utf-8')) for fold in range(3)]


def selected(reports):
    return np.array([q['primary'] for report in reports for q in report['queries']])


def last(reports):
    return np.array([v for report in reports for v in report['last_epoch']['primary']])


def interval(values, rows):
    report = bootstrap_gap(values, rows, 10000, 42, level=95)
    return {'estimate': _macro(values, rows), 'reader_interval': report['reader_interval'], 'sentence_interval': report['sentence_interval']}


def main():
    if OUTPUT.exists():
        raise FileExistsError('Use a fresh output directory')
    plan = json.loads((SOURCE/'plan.json').read_text(encoding='utf-8'))['plan']
    records = [r for path in plan['cohort']['manifests'] for r in read_manifest(ROOT/path)]
    layout = plan['split']
    excluded = excluded_positions(records, plan['seed'], **layout['v2_test_zone'])
    folds = proximity_folds(records, excluded, layout['cycle_sentences'], layout['cluster_sentences'], layout['fold_offsets'],
                            layout['near_lag'], layout['gap_lags'], plan['seed'])
    forbidden = {(release, i) for release, ids in excluded.items() for i in ids}
    allowed = [r for r in records if (r.dataset, int(r.sentence_id)) not in forbidden]
    by_fold = [rows_for(allowed, roles['score']) for roles in folds]
    rows = [r for fold in by_fold for r in fold]
    fold_of = np.array([i for i, fold in enumerate(by_fold) for _ in fold])
    release = np.array([r.dataset for r in rows])
    identity = [q['identity'] for report in load('cnn_F_full_near_real') for q in report['queries']]
    if [(r.dataset, r.subject_key.split(':')[-1], str(r.sentence_id)) for r in rows] != [(q[0], q[1], str(q[4])) for q in identity]:
        raise ValueError('Saved queries do not line up with the rebuilt rows')
    cnn = {arm: load(f'cnn_{arm}') for arm in CNN_ARMS}
    ridge = {arm: load(f'ridge_{arm}') for arm in CNN_ARMS}
    result = {'note': 'Exploratory, chosen after seeing the v3 result. Reused sentences, no accuracy claim.', 'cnn': {}, 'ridge': {}}
    for arm, reports in cnn.items():
        result['cnn'][arm] = {'selected_10_way': _macro(selected(reports), rows), 'last_10_way': _macro(last(reports), rows),
                              'best_epochs': [r['best_epoch'] for r in reports], 'epochs_run': [r['epochs_run'] for r in reports],
                              'fold_mean': {w: float(np.mean([r['metrics'][f'{w}_way'] for r in reports])) for w in (2, 4, 10)},
                              'fold_mean_last': {w: float(np.mean([r['last_epoch']['metrics'][f'{w}_way'] for r in reports])) for w in (2, 4, 10)}}
    for arm, reports in ridge.items():
        result['ridge'][arm] = {'selected_10_way': _macro(selected(reports), rows), 'alphas': [r['selected_alpha'] for r in reports],
                                'fold_mean': {w: float(np.mean([r['metrics'][f'{w}_way'] for r in reports])) for w in (2, 4, 10)}}
    for name in ('timing', 'position', 'text'):
        for condition in ('near', 'far'):
            reports = load(f'{name}_{condition}')
            result[f'{name}_{condition}'] = {'selected_10_way': _macro(selected(reports), rows),
                                             'fold_mean': {w: float(np.mean([r['metrics'][f'{w}_way'] for r in reports])) for w in (2, 4, 10)}}
    near, far = last(cnn['F_full_near_real']), last(cnn['F_full_far_real'])
    noise_near, noise_far = last(cnn['F_full_near_gaussian']), last(cnn['F_full_far_gaussian'])
    shuffled_near, shuffled_far = last(cnn['F_full_near_shuffled']), last(cnn['F_full_far_shuffled'])
    result['cnn_last_epoch'] = {
        'gain_real': interval(near - far, rows), 'gain_noise': interval(noise_near - noise_far, rows), 'gain_shuffled': interval(shuffled_near - shuffled_far, rows),
        'gain_real_minus_noise': interval((near - far) - (noise_near - noise_far), rows),
        'near_real_minus_near_noise': interval(near - noise_near, rows), 'near_real_minus_near_shuffled': interval(near - shuffled_near, rows),
        'far_real_minus_far_noise': interval(far - noise_far, rows), 'far_real_minus_far_shuffled': interval(far - shuffled_far, rows)}
    r_near, r_far = selected(ridge['F_full_near_real']), selected(ridge['F_full_far_real'])
    r_noise_near, r_noise_far = selected(ridge['F_full_near_gaussian']), selected(ridge['F_full_far_gaussian'])
    result['ridge_comparisons'] = {
        'gain_noise': interval(r_noise_near - r_noise_far, rows), 'gain_real_minus_noise': interval((r_near - r_far) - (r_noise_near - r_noise_far), rows),
        'near_real_minus_near_noise': interval(r_near - r_noise_near, rows), 'near_real_minus_near_shuffled': interval(r_near - selected(ridge['F_full_near_shuffled']), rows)}
    result['gain_by_fold'] = {}
    result['gain_by_release'] = {}
    for label, values in (('cnn_selected', selected(cnn['F_full_near_real']) - selected(cnn['F_full_far_real'])), ('cnn_last', near - far), ('ridge', r_near - r_far)):
        result['gain_by_fold'][label] = [_macro(values[fold_of == i], [r for r, f in zip(rows, fold_of) if f == i]) for i in range(3)]
        result['gain_by_release'][label] = {name: _macro(values[release == name], [r for r in rows if r.dataset == name]) for name in sorted(set(release))}
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT/'details.json', result)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
