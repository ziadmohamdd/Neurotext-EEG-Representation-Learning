"""Audit wider-grid exploratory repeats; describe split variation, not new cohorts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

from ..utils.logging import write_json

ALPHAS = [.1, 1., 10., 100., 1000., 10000., 100000.]
DIRECTIONS = ['zuco1_to_zuco2', 'zuco2_to_zuco1']
CONTROLS = ['real', 'gaussian', 'shuffled']
SEEDS = [42, 43, 44]


def read_run(path):
    path = Path(path)
    report = json.loads(path.read_text(encoding='utf-8'))
    meta = json.loads((path.parent/'metadata.json').read_text(encoding='utf-8'))
    config, split = meta['config'], meta['splits']
    source, target = meta['train_datasets'], meta['test_datasets']
    if len(source) != 1 or len(target) != 1:
        raise ValueError('Each run must have one source and one target release')
    direction = source[0]+'_to_'+target[0]
    control, seed = report['training_control'], report['seed']
    if (direction not in DIRECTIONS or control not in CONTROLS or seed not in SEEDS
            or meta['training_control'] != control or config['seed'] != seed or split['seed'] != seed):
        raise ValueError('Unexpected direction, control or seed metadata')
    if (report['status'] != 'executed' or meta['stage'] != 'ridge'
            or report['protocol'] != 'transfer_exploratory' or split['protocol'] != 'transfer_exploratory'
            or config.get('ridge_alphas') != ALPHAS or config.get('max_samples') is not None
            or config.get('quality_aware', False)):
        raise ValueError('Require completed full-recording, wider-grid exploratory statistical ridge')
    if meta.get('encoder_origin') or report.get('corruption'):
        raise ValueError('Pretrained features or corruption do not belong in this comparison')
    if report['evaluation_control'] != ('gaussian' if control == 'gaussian' else 'real'):
        raise ValueError('Evaluation control differs from the matched training condition')
    if report['train_datasets'] != source or report['test_datasets'] != target:
        raise ValueError('Training/evaluation release metadata disagree')
    parts = split['splits']
    overlap = len((set(parts['train']['sentence_keys']) | set(parts['val']['sentence_keys']))
                  & set(parts['test']['sentence_keys']))
    for assessment in [meta['transfer_assessment'], report['transfer_assessment']]:
        if assessment['identity_audited'] is not False or assessment['sentence_overlap'] != overlap:
            raise ValueError('Exploratory identity or shared-text disclosure changed')
    subjects = {name:set(parts[name]['subjects']) for name in ['train', 'val', 'test']}
    if (any(not s for s in subjects.values()) or subjects['train'] & subjects['val']
            or (subjects['train'] | subjects['val']) & subjects['test']
            or set(meta['normalizer']['subjects']) != subjects['train']):
        raise ValueError('Normalizer or subject splits do not use source training only')
    for name in ['train', 'val', 'test']:
        expected = source[0] if name != 'test' else target[0]
        if not parts[name]['observations'] or any(obs[0] != expected for obs in parts[name]['observations']):
            raise ValueError('Source/target observations entered the wrong split')
    selection = meta['selection']
    if [r['alpha'] for r in selection] != ALPHAS:
        raise ValueError('Validation selection table does not match the fixed wider grid')
    if any(not 0 < r['validation_mrr'] <= 1 for r in selection):
        raise ValueError('Invalid validation MRR')
    winner = max(selection, key=lambda r:r['validation_mrr'])  # first wins ties, as in training
    if meta['selected_alpha'] != winner['alpha']:
        raise ValueError('Selected alpha does not maximize source-validation MRR')
    pool = report['candidate_keys']
    digest = hashlib.sha256(json.dumps(pool).encode()).hexdigest()
    if (len(pool) != len(set(pool)) or pool != meta['candidate_keys']
            or pool != parts['test']['sentence_keys'] or digest != report['candidate_pool_sha256']):
        raise ValueError('Candidate pool keys or hash disagree')
    metrics = report['metrics']
    if len(parts['test']['observations']) != metrics['n_queries']:
        raise ValueError('Query count differs from saved target split')
    if any(not 0 <= metrics[k] <= 1 for k in ['recall@1', 'mrr']):
        raise ValueError('Invalid evaluation metrics')
    row = dict(direction=direction, seed=seed, control=control,
               selected_alpha=meta['selected_alpha'], validation_mrr=winner['validation_mrr'],
               recall_at_1=metrics['recall@1'], mrr=metrics['mrr'],
               n_queries=metrics['n_queries'], n_candidates=len(pool),
               shared_source_target_texts=overlap,
               candidate_pool_sha256=digest, text_cache_digest=meta['text_cache_digest'],
               train_digest=parts['train']['digest'], val_digest=parts['val']['digest'],
               test_digest=parts['test']['digest'], protocol=report['protocol'], identity_audited=False,
               report=str(path.resolve()))
    audit = dict(direction=direction, seed=seed, control=control, config=config,
                 source_training_subjects=parts['train']['subjects'],
                 source_validation_subjects=parts['val']['subjects'],
                 target_subjects=parts['test']['subjects'], selection=selection,
                 validation_candidate_keys=parts['val']['sentence_keys'],
                 candidate_keys=pool, normalizer=meta['normalizer'],
                 report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 metadata_sha256=hashlib.sha256((path.parent/'metadata.json').read_bytes()).hexdigest())
    return row, audit


def aggregate_runs(rows):
    indexed = {(r['direction'], r['seed'], r['control']):r for r in rows}
    expected = {(d,s,c) for d in DIRECTIONS for s in SEEDS for c in CONTROLS}
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError('Require exactly one wider-grid run per direction, seed and control (18 total)')
    # Same target queries and texts across controls AND seeds, not independent cohorts.
    for direction in DIRECTIONS:
        group = [r for r in rows if r['direction'] == direction]
        for field in ['candidate_pool_sha256', 'test_digest', 'n_queries', 'n_candidates', 'text_cache_digest', 'shared_source_target_texts']:
            if len({r[field] for r in group}) != 1:
                raise ValueError(f'Mismatched {field} across repeats/controls')
        for seed in SEEDS:
            group = [indexed[direction,seed,c] for c in CONTROLS]
            for field in ['train_digest', 'val_digest']:
                if len({r[field] for r in group}) != 1:
                    raise ValueError(f'Mismatched source {field} within seed')
    condition, paired, differences = [], [], []
    for direction in DIRECTIONS:
        for control in CONTROLS:
            group = [indexed[direction,seed,control] for seed in SEEDS]
            condition.append(dict(direction=direction, control=control,
                                  mean_mrr=mean(r['mrr'] for r in group),
                                  split_sd_mrr=stdev(r['mrr'] for r in group),
                                  mean_recall_at_1=mean(r['recall_at_1'] for r in group),
                                  split_sd_recall_at_1=stdev(r['recall_at_1'] for r in group)))
        for seed in SEEDS:
            real, noise = indexed[direction,seed,'real'], indexed[direction,seed,'gaussian']
            paired.append(dict(direction=direction, seed=seed,
                               real_minus_gaussian_mrr=real['mrr']-noise['mrr'],
                               real_minus_gaussian_recall_at_1_pp=100*(real['recall_at_1']-noise['recall_at_1'])))
        group = [p for p in paired if p['direction'] == direction]
        differences.append(dict(direction=direction,
                                mean_mrr_difference=mean(p['real_minus_gaussian_mrr'] for p in group),
                                split_sd_mrr_difference=stdev(p['real_minus_gaussian_mrr'] for p in group),
                                mean_recall_at_1_difference_pp=mean(p['real_minus_gaussian_recall_at_1_pp'] for p in group),
                                split_sd_recall_at_1_difference_pp=stdev(p['real_minus_gaussian_recall_at_1_pp'] for p in group),
                                positive_mrr_seeds=sum(p['real_minus_gaussian_mrr']>0 for p in group)))
    return dict(runs=sorted(rows,key=lambda r:(r['direction'],r['seed'],r['control'])),
                condition_summary=condition, paired_differences=paired, difference_summary=differences,
                interpretation='Equal-weight means and sample SD across three source splits/control seeds; '
                'the same target participants and queries are reused. SD is descriptive, not a confidence '
                'interval. Repeated splits are not independent cohorts; no pooled sample size or significance claim.')


def summarize_roots(roots, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a fresh summary output directory')
    rows, audits = [], []
    for root in roots:
        for path in sorted(Path(root).rglob('evaluation.json')):
            row, audit = read_run(path)
            rows.append(row)
            audits.append(audit)
    summary = aggregate_runs(rows)
    # Recipes must be identical apart from the declared experimental seed.
    recipes = [{k:v for k,v in a['config'].items() if k != 'seed'} for a in audits]
    if any(recipe != recipes[0] for recipe in recipes):
        raise ValueError('Effective configurations differ beyond seed')
    for direction in DIRECTIONS:
        for seed in SEEDS:
            group = [a for a in audits if a['direction']==direction and a['seed']==seed]
            if any(a['normalizer'] != group[0]['normalizer'] for a in group):
                raise ValueError('Control normalizers differ within a source split')
    output.mkdir(parents=True)
    write_json(output/'summary.json', summary)
    write_json(output/'configurations_and_pools.json', audits)
    for name, records in [('runs',summary['runs']),('condition_summary',summary['condition_summary']),
                          ('paired_differences',summary['paired_differences']),('difference_summary',summary['difference_summary'])]:
        with (output/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    summary = summarize_roots(args.roots,args.output)
    print(json.dumps({k:v for k,v in summary.items() if k!='runs'},indent=2))


if __name__ == '__main__':
    main()
