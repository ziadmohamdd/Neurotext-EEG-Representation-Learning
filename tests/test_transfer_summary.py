"""Synthetic reports test auditing/statistics only, not scientific retrieval."""
import copy
import hashlib
import json
from statistics import stdev

import pytest

from eeg_decoding.evaluation.transfer_summary import (
    ALPHAS, CONTROLS, DIRECTIONS, SEEDS, aggregate_runs, read_run, summarize_roots,
)


@pytest.fixture
def report_roots(tmp_path):
    root = tmp_path/'SYNTHETIC_reports'
    for direction in DIRECTIONS:
        source,target = direction.split('_to_')
        for seed in SEEDS:
            for control in CONTROLS:
                directory = root/f'{direction}_{seed}_{control}'
                directory.mkdir(parents=True)
                parts = {}
                for name,dataset,subject in [('train',source,'a'),('val',source,'b'),('test',target,'c')]:
                    parts[name] = dict(subjects=[dataset+':'+subject],sentence_keys=['k1','k2'],
                                       observations=[[dataset,subject,'1','nr',str(i)] for i in range(2)],
                                       digest=dataset+name+(str(seed) if name!='test' else 'fixed'))
                assessment = dict(identity_audited=False,sentence_overlap=2)
                meta = dict(stage='ridge',train_datasets=[source],test_datasets=[target],
                            config=dict(seed=seed,ridge_alphas=ALPHAS,max_samples=None),
                            splits=dict(seed=seed,protocol='transfer_exploratory',splits=parts),
                            training_control=control,normalizer=dict(subjects=parts['train']['subjects']),
                            transfer_assessment=assessment,candidate_keys=['k1','k2'],
                            text_cache_digest='SYNTHETIC_fixed',selected_alpha=100.,
                            selection=[dict(alpha=a,validation_mrr=.5 if a==100 else .2) for a in ALPHAS])
                score = {'real':[.1,.2,.3], 'gaussian':[.2,.1,.2], 'shuffled':[.15,.15,.15]}[control][seed-42]
                report = dict(status='executed',training_control=control,seed=seed,
                              protocol='transfer_exploratory',train_datasets=[source],test_datasets=[target],
                              evaluation_control='gaussian' if control=='gaussian' else 'real',
                              transfer_assessment=assessment,candidate_keys=['k1','k2'],
                              candidate_pool_sha256=hashlib.sha256(json.dumps(['k1','k2']).encode()).hexdigest(),
                              metrics={'n_queries':2,'recall@1':0.,'mrr':score})
                (directory/'metadata.json').write_text(json.dumps(meta))
                (directory/'evaluation.json').write_text(json.dumps(report))
    return root


def test_three_seed_summary_pairs_seeds_and_uses_sample_sd(report_roots,tmp_path):
    before = {p:p.read_bytes() for p in report_roots.rglob('*.json')}
    result = summarize_roots([report_roots],tmp_path/'summary')
    assert before == {p:p.read_bytes() for p in report_roots.rglob('*.json')}
    assert len(result['runs']) == 18
    real = next(r for r in result['condition_summary'] if r['control']=='real')
    assert real['mean_mrr'] == pytest.approx(.2)
    assert real['split_sd_mrr'] == pytest.approx(.1)
    differences = result['difference_summary'][0]
    assert differences['mean_mrr_difference'] == pytest.approx(1/30)
    assert differences['split_sd_mrr_difference'] == pytest.approx(stdev([-.1,.1,.1]))
    assert differences['positive_mrr_seeds'] == 2
    assert 'not independent cohorts' in result['interpretation']
    assert 'ci95' not in json.dumps(result)
    with pytest.raises(FileExistsError):
        summarize_roots([report_roots],tmp_path/'summary')


def test_summary_rejects_missing_duplicate_or_incomparable_runs(report_roots):
    rows = [read_run(p)[0] for p in report_roots.rglob('evaluation.json')]
    for invalid in [rows[:-1],rows+[rows[0]]]:
        with pytest.raises(ValueError,match='18 total'):
            aggregate_runs(invalid)
    for field in ['candidate_pool_sha256','test_digest','train_digest']:
        changed = copy.deepcopy(rows)
        changed[0][field] = 'different'
        with pytest.raises(ValueError,match='Mismatched'):
            aggregate_runs(changed)


@pytest.mark.parametrize('change', ['alpha','identity','normalizer','target_in_train','pool_hash','grid'])
def test_summary_rejects_wrong_selection_or_protocol(report_roots,change):
    path = next(report_roots.rglob('evaluation.json'))
    meta_path = path.parent/'metadata.json'
    meta = json.loads(meta_path.read_text())
    report = json.loads(path.read_text())
    if change=='alpha':
        meta['selected_alpha'] = 100000.
    elif change=='identity':
        report['transfer_assessment']['identity_audited'] = True
    elif change=='normalizer':
        meta['normalizer']['subjects'] = meta['splits']['splits']['val']['subjects']
    elif change=='target_in_train':
        meta['splits']['splits']['train']['observations'][0][0] = meta['test_datasets'][0]
    elif change=='pool_hash':
        report['candidate_pool_sha256'] = 'bad'
    elif change=='grid':
        meta['config']['ridge_alphas'] = [.1,1,10,100]
    meta_path.write_text(json.dumps(meta))
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        read_run(path)
