"""Rebuild descriptive tables/figures from the archived user-supplied evaluation CSV."""
import argparse
import ast
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def equivalent_report(rows):
    """Select a display representative only when every exported value agrees."""
    if not rows:
        raise ValueError('Required evaluation condition is missing')
    signature = lambda row: {k:v for k,v in row.items() if k not in ('file','name')}
    if any(signature(row) != signature(rows[0]) for row in rows[1:]):
        raise ValueError('Repeated condition has differing exported values; explicit run review required')
    return sorted(rows,key=lambda row:row['name'])[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='docs/research_results/retrieval.csv')
    parser.add_argument('--output', default='docs/research_results')
    args = parser.parse_args()
    source, output = Path(args.input), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with source.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    files = {}
    repeated = defaultdict(list)
    for row in rows:
        row['name'] = row['file'].replace('\\', '/').rsplit('/', 1)[-1]
        if row['name'] in files:
            raise ValueError('Repeated file name in export: '+row['name'])
        files[row['name']] = row
        for key in ['recall@1','recall@5','recall@10','mrr']:
            value = float(row[key])
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Invalid metric: '+row['name'])
        # Equal aggregate values are identified, never treated as proof of duplicate artifacts.
        signature = tuple((k,v) for k,v in row.items() if k not in ('file','name'))
        repeated[signature].append(row['name'])
    repeated = [names for names in repeated.values() if len(names)>1]
    summary = []
    fig, axes = plt.subplots(1,2, figsize=(10,4), sharey=True, constrained_layout=True)
    for ax, dataset, candidates in zip(axes, ['zuco1','zuco2'], [60,68]):
        chance = sum(1/i for i in range(1,candidates+1))/candidates
        for index, control in enumerate(['real','gaussian','shuffled']):
            selected = [r for r in rows if r['name'].startswith(dataset+'_')
                        and r['protocol']=='subject_sentence' and r['training_control']==control
                        and r['corruption']=='None']
            if sorted(int(r['seed']) for r in selected) != [42,43,44]:
                raise ValueError(f'Require exactly seeds 42–44 for {dataset}/{control}')
            selected.sort(key=lambda r:int(r['seed']))
            values = np.array([float(r['mrr']) for r in selected])
            summary.append(dict(dataset=dataset, control=control, seeds='42;43;44',
                                mean_recall1=np.mean([float(r['recall@1']) for r in selected]),
                                mean_mrr=values.mean(), sample_sd_mrr=values.std(ddof=1),
                                chance_mrr=chance, query_counts=';'.join(r['n_queries'] for r in selected)))
            ax.scatter(np.full(3,index)+[-.08,0,.08], values, s=32)
            ax.plot([index-.16,index+.16], [values.mean()]*2, color='black', lw=2)
        ax.axhline(chance, color='gray', ls='--', label='Random-ranking expectation')
        ax.set(title=f'{dataset.upper()} · {candidates} candidates', xticks=[0,1,2],
               xticklabels=['Real EEG','Gaussian','Shuffled'], ylim=(0,.12))
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Mean reciprocal rank (MRR)')
    axes[1].legend(loc='lower right', fontsize=8)
    fig.suptitle('Unseen subjects and sentences: individual splits and three-split means')
    fig.savefig(output/'unseen_content_mrr.png', dpi=200)
    fig.savefig(output/'unseen_content_mrr.svg')
    plt.close(fig)

    fig, axes = plt.subplots(1,2, figsize=(9,4), constrained_layout=True)
    for ax, dataset in zip(axes, ['zuco1','zuco2']):
        values = []
        for prefix in ['statistical19','labram19']:
            selected = [r for r in rows if r['name'].startswith(f'{dataset}_{prefix}_')
                        and r['training_control']=='real' and r['evaluation_control']=='real'
                        and r['corruption']=='None']
            if len(selected) != 1:
                raise ValueError('Ambiguous matched representation report')
            values.append(float(selected[0]['mrr']))
        ax.bar(['Statistical + ridge','LaBraM + ridge'], values, color=['#386cb0','#f28e2b'])
        n = 300 if dataset=='zuco1' else 344
        ax.axhline(sum(1/i for i in range(1,n+1))/n, color='gray', ls='--')
        ax.set(title=f'{dataset.upper()} · {n} candidates', ylabel='MRR', ylim=(0,.035))
    fig.suptitle('Matched 19-channel inputs · one subject split per dataset')
    fig.savefig(output/'pretrained_comparison.png', dpi=200)
    fig.savefig(output/'pretrained_comparison.svg')
    plt.close(fig)

    labram_summary = []
    fig, axes = plt.subplots(1,2, figsize=(9,4), constrained_layout=True)
    for ax, dataset, n in zip(axes, ['zuco1','zuco2'], [300,344]):
        baseline = equivalent_report([r for r in rows if r['name'].startswith(dataset+'_labram19_')
                                     and r['training_control']=='real' and r['corruption']=='None'])
        values = []
        for control in ['real','gaussian','shuffled']:
            selected = [r for r in rows if r['name'].startswith(dataset+'_labram19_')
                        and r['training_control']==control and r['corruption']=='None']
            row = equivalent_report(selected)
            for key in ['protocol','seed','candidate_pool_sha256','n_queries','quality_aware']:
                if row[key] != baseline[key]:
                    raise ValueError(f'LaBraM comparison mismatch: {dataset}/{control}/{key}')
            if row['evaluation_control'] != ('gaussian' if control=='gaussian' else 'real'):
                raise ValueError('LaBraM evaluation control mismatch')
            values.append(float(row['mrr']))
            labram_summary.append(dict(dataset=dataset, condition=control, reports_with_equal_values=len(selected),
                                       source_files=';'.join(sorted(r['file'] for r in selected)),
                                       seed=row['seed'], n_queries=row['n_queries'], candidates=n,
                                       **{key:row[key] for key in ['recall@1','recall@5','recall@10','mrr']}))
        ax.bar(['Real EEG','Gaussian','Shuffled'], values, color=['#386cb0','#f28e2b','#59a14f'])
        ax.axhline(sum(1/i for i in range(1,n+1))/n,color='gray',ls='--')
        ax.set(title=f'{dataset.upper()} · {n} candidates', ylabel='MRR', ylim=(0,.035))
    fig.suptitle('Frozen LaBraM + ridge controls · seed 42, matched 19-channel cohorts')
    fig.savefig(output/'labram_controls_mrr.png', dpi=200)
    svg = output/'labram_controls_mrr.svg'
    fig.savefig(svg, metadata={'Date':None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)
    with (output/'labram_controls_summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=list(labram_summary[0]))
        writer.writeheader()
        writer.writerows(labram_summary)

    cnn_summary = []
    fig, axes = plt.subplots(1,2, figsize=(9,4), constrained_layout=True)
    for ax, dataset, n, original, prefix in [
        (axes[0], 'zuco1', 300, 'zuco1_cnn_20261027_131425.json', 'zuco1_cnn_original_'),
        (axes[1], 'zuco2', 344, 'zuco2_cnn_full.json', 'zuco2_cnn_full_'),
    ]:
        baseline = files[original]
        values = []
        for control in ['real','gaussian','shuffled']:
            selected = [baseline] if control=='real' else [
                r for r in rows if r['name'].startswith(prefix+control+'_')]
            if len(selected) != 1:
                raise ValueError(f'Require one CNN report for {dataset}/{control}')
            row = selected[0]
            for key in ['protocol','seed','candidate_pool_sha256','n_queries','quality_aware']:
                if row[key] != baseline[key]:
                    raise ValueError(f'CNN comparison mismatch: {dataset}/{control}/{key}')
            if row['training_control']!=control or row['evaluation_control']!=('gaussian' if control=='gaussian' else 'real'):
                raise ValueError('CNN control provenance mismatch')
            values.append(float(row['mrr']))
            cnn_summary.append(dict(dataset=dataset, condition=control, file=row['file'],
                                    seed=row['seed'], n_queries=row['n_queries'], candidates=n,
                                    **{key:row[key] for key in ['recall@1','recall@5','recall@10','mrr']}))
        ax.bar(['Real EEG','Gaussian','Shuffled'], values, color=['#386cb0','#f28e2b','#59a14f'])
        ax.axhline(sum(1/i for i in range(1,n+1))/n, color='gray', ls='--')
        recipe = 'Original cohort, 8-second crop' if dataset=='zuco1' else 'Duration QC, full recordings'
        ax.set(title=f'{dataset.upper()} · {n} candidates\n{recipe}', ylabel='MRR', ylim=(0,.03))
    fig.suptitle('Supervised CNN controls · seed 42, held-out subjects')
    fig.savefig(output/'cnn_controls_mrr.png', dpi=200)
    fig.savefig(output/'cnn_controls_mrr.svg', metadata={'Date':None})
    svg = output/'cnn_controls_mrr.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
    plt.close(fig)
    with (output/'cnn_controls_summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=list(cnn_summary[0]))
        writer.writeheader()
        writer.writerows(cnn_summary)

    ssl_summary = []
    fig, axes = plt.subplots(1,2, figsize=(9,4), constrained_layout=True)
    for ax, dataset, n, original in [
        (axes[0], 'zuco1', 300, 'zuco1_full_sentence_alignment.json'),
        (axes[1], 'zuco2', 344, 'zuco2_ssl_alignment_full.json'),
    ]:
        baseline = files[original]
        values = []
        for control in ['real','gaussian','shuffled']:
            selected = [baseline] if control=='real' else [r for r in rows
                        if r['name'].startswith(f'{dataset}_ssl_alignment_{control}_')]
            row = equivalent_report(selected)
            for key in ['protocol','seed','candidate_pool_sha256','n_queries','quality_aware']:
                if row[key] != baseline[key]:
                    raise ValueError(f'SSL comparison mismatch: {dataset}/{control}/{key}')
            if row['training_control']!=control or row['evaluation_control']!=('gaussian' if control=='gaussian' else 'real'):
                raise ValueError('SSL control provenance mismatch')
            values.append(float(row['mrr']))
            ssl_summary.append(dict(dataset=dataset, condition=control,
                                    source_files=';'.join(sorted(r['file'] for r in selected)),
                                    seed=row['seed'], n_queries=row['n_queries'], candidates=n,
                                    **{key:row[key] for key in ['recall@1','recall@5','recall@10','mrr']}))
        ax.bar(['Real EEG','Gaussian','Shuffled'], values, color=['#386cb0','#f28e2b','#59a14f'])
        ax.axhline(sum(1/i for i in range(1,n+1))/n,color='gray',ls='--')
        ax.set(title=f'{dataset.upper()} · {n} candidates', ylabel='MRR', ylim=(0,.03))
    fig.suptitle('SSL + alignment controls · full recordings, seed 42, held-out subjects')
    fig.savefig(output/'ssl_controls_mrr.png', dpi=200)
    svg = output/'ssl_controls_mrr.svg'
    fig.savefig(svg, metadata={'Date':None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)
    with (output/'ssl_controls_summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=list(ssl_summary[0]))
        writer.writeheader()
        writer.writerows(ssl_summary)

    fig, axes = plt.subplots(1,3, figsize=(12,3.8), sharey=True, constrained_layout=True)
    for ax, kind, label in zip(axes, ['additive_noise','channel_dropout','temporal_mask'],
                               ['Added noise / channel SD','Fraction of channels dropped','Fraction of time masked']):
        for quality, display in [('False','Standard ridge'),('True','Quality-aware ridge')]:
            points = []
            for row in rows:
                if row['corruption']=='None' or row['quality_aware']!=quality:
                    continue
                corruption = ast.literal_eval(row['corruption'])
                if row['name'].startswith('zuco2_') and corruption[0]==kind:
                    points.append((float(corruption[1]),float(row['mrr'])))
            points.sort()
            if len(points) != 4 or len({x for x,y in points})!=4:
                raise ValueError('Require four distinct severities per condition')
            ax.plot(*zip(*points), marker='o', label=display)
        ax.axhline(sum(1/i for i in range(1,345))/344, color='gray', ls='--')
        ax.set(xlabel=label, title=kind.replace('_',' ').capitalize(), ylim=(0,.03))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('MRR')
    axes[2].legend(fontsize=8, loc='lower left')
    fig.suptitle('ZuCo2 robustness · one subject split and one corruption seed')
    fig.savefig(output/'quality_robustness_mrr.png', dpi=200)
    fig.savefig(output/'quality_robustness_mrr.svg')
    plt.close(fig)
    with (output/'unseen_content_summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    audit = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), n_reports=len(rows),
                 identical_export_value_groups=repeated,
                 warning='Equal exported values do not prove identical checkpoints. No rows were deleted.',
                 provenance='User-supplied real-data CSV; original JSON/checkpoints remain on user machine.',
                 missing_export_fields=['artifact identity','signal/split digests','config/cropping','candidate count',
                                        'subject-wise metrics','confidence intervals'],
                 candidate_counts_source='Previously supplied evaluation console output, not present in CSV')
    (output/'export_audit.json').write_text(json.dumps(audit,indent=2), encoding='utf-8')
    print(f'Summarized {len(rows)} reports; {len(repeated)} groups have identical exported values.')
    print('Generated six figure pairs, four summary CSVs and export_audit.json in',output)


if __name__=='__main__':
    main()
