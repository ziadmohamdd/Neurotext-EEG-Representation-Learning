"""Archive and plot existing wider-grid results. Never trains or evaluates a model."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from eeg_decoding.evaluation.transfer_summary import read_run, aggregate_runs

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'results/exploratory_three_seed_summary_20261029'
OUT = ROOT/'docs/research_results/wider_grid_transfer'
ARCHIVE = ROOT/'docs/presentation/archive/pre_wider_grid'
COLORS = ['#386cb0', '#f28e2b', '#59a14f']
CONTROLS = ['real', 'gaussian', 'shuffled']
DIRECTIONS = ['zuco1_to_zuco2', 'zuco2_to_zuco1']
LABELS = ['ZuCo1 → ZuCo2', 'ZuCo2 → ZuCo1']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists() or ARCHIVE.exists():
        raise FileExistsError('Consolidation archive exists; do not overwrite provenance')
    summary = json.loads((SOURCE/'summary.json').read_text(encoding='utf-8'))
    audited = [read_run(r['report'])[0] for r in summary['runs']]
    if aggregate_runs(audited) != summary:
        raise ValueError('Stored summary disagrees with archived evaluation reports')
    OUT.mkdir(parents=True)
    ARCHIVE.mkdir(parents=True)
    for name in ['NeuroText_draft.pptx', 'NeuroText_draft.pdf', 'README.md']:
        shutil.copy2(ROOT/'docs/presentation'/name, ARCHIVE/name)
    for source in ['docs/research_results/REPORT.md', 'docs/presentation_outline.md', 'scripts/build_presentation.py']:
        shutil.copy2(ROOT/source, ARCHIVE/Path(source).name)
    ledger = []

    def copy(source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        ledger.append(dict(source=str(source.relative_to(ROOT)),
                           archived=str(destination.relative_to(OUT)),sha256=sha(source)))

    for source in sorted(SOURCE.iterdir()):
        if source.is_file():
            copy(source, OUT/source.name)
    for name in ['transfer_ridge_regularized.yaml', 'labram_full_sentence.yaml', 'local.yaml',
                 'transfer_exploratory_ridge.yaml', 'transfer_exploratory_seed43.yaml',
                 'transfer_exploratory_seed44.yaml']:
        copy(ROOT/'configs'/name,OUT/'configs'/name)
    for source in ['src/eeg_decoding/models/classical.py', 'src/eeg_decoding/workflow.py',
                   'src/eeg_decoding/evaluation/transfer_summary.py', 'scripts/refit_ridge_path.py']:
        copy(ROOT/source,OUT/'implementation'/Path(source).name)
    for r in summary['runs']:
        name=f"{r['direction']}_s{r['seed']}_{r['control']}.json"
        copy(Path(r['report']),OUT/'evaluations'/name)
    # Keep the original four-alpha seed-42 transfer comparison separate.
    original=[]
    for path in sorted((ROOT/'results/exploratory_transfer_20261029_232616').glob('*/evaluation.json')):
        r=json.loads(path.read_text())
        m=json.loads((path.parent/'metadata.json').read_text())
        original.append(dict(grid='original_0.1_to_100',seed=r['seed'],
                             direction=r['train_datasets'][0]+'_to_'+r['test_datasets'][0],
                             control=r['training_control'],selected_alpha=m['selected_alpha'],
                             recall_at_1=r['metrics']['recall@1'],mrr=r['metrics']['mrr']))
        copy(path,OUT/'original_grid_evaluations'/path.parent.name/'evaluation.json')
    with (OUT/'original_grid_seed42.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(original[0])); writer.writeheader(); writer.writerows(original)
    manifest=dict(scope='18 wider-grid exploratory evaluations; original-grid seed42 stored separately',
                  no_new_experiments=True,source_summary_sha256=sha(SOURCE/'summary.json'),
                  alpha_grid=[.1,1,10,100,1000,10000,100000],
                  validation=dict(passed=139,optional_skips=2,deselected=0,
                      command='.venv/Scripts/python.exe -m pytest -q -rs',
                      skips=[dict(test='tests/test_storage_and_pretrained.py::test_official_pretrained_weights_on_synthetic_input',
                                  reason='optional official-weight integration; no network download during unit tests'),
                             dict(test='tests/test_storage_and_pretrained.py::test_pretrained_cache_extraction',
                                  reason='optional official-weight extraction on synthetic fixtures')],
                      skip_condition='EEG_LABRAM_CACHE unset; large-MAT test included and passed'),
                  files=ledger)
    (OUT/'archive_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

    runs=summary['runs']
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes=plt.subplots(1,2,figsize=(11,4.3),sharey=True,layout='constrained')
    for ax,direction,label,n in zip(axes,DIRECTIONS,LABELS,[344,300]):
        for c,color in zip(CONTROLS,COLORS):
            values=[next(r['mrr'] for r in runs if (r['direction'],r['seed'],r['control'])==(direction,s,c)) for s in [42,43,44]]
            ax.plot([42,43,44],values,'o-',label=c.capitalize(),color=color,lw=1.6)
        ax.axhline(sum(1/i for i in range(1,n+1))/n,ls='--',color='#666666',lw=1,label='Uniform ranking')
        ax.set(title=f'{label} · {n} candidates',xticks=[42,43,44],xlabel='Seed',ylim=(.015,.028))
        ax.grid(axis='y',alpha=.18)
    axes[0].set_ylabel('Mean reciprocal rank (MRR)'); axes[1].legend(fontsize=9)
    fig.suptitle('Wider grid: seed-level exploratory transfer')
    for ext in ['png','svg']: fig.savefig(OUT/f'wider_grid_seed_mrr.{ext}',dpi=200)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.3),sharey=True,layout='constrained')
    for ax,direction,label,n in zip(axes,DIRECTIONS,LABELS,[344,300]):
        group=[next(r for r in summary['condition_summary'] if r['direction']==direction and r['control']==c) for c in CONTROLS]
        ax.bar(range(3),[r['mean_mrr'] for r in group],yerr=[r['split_sd_mrr'] for r in group],capsize=5,color=COLORS,alpha=.8)
        for i,c in enumerate(CONTROLS):
            values=[next(r['mrr'] for r in runs if (r['direction'],r['seed'],r['control'])==(direction,s,c)) for s in [42,43,44]]
            ax.scatter(np.array([-.08,0,.08])+i,values,color='#142D44',s=24,zorder=4)
        ax.axhline(sum(1/i for i in range(1,n+1))/n,ls='--',color='#666666',lw=1)
        ax.set(title=label,xticks=range(3),xticklabels=['Real EEG','Gaussian','Shuffled'],ylim=(0,.029))
        ax.grid(axis='y',alpha=.18)
    axes[0].set_ylabel('MRR · equal-weight mean ± sample SD')
    fig.suptitle('Wider grid: three repeated splits, same target cohort')
    for ext in ['png','svg']: fig.savefig(OUT/f'wider_grid_mean_sd.{ext}',dpi=200)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.3),sharey=True,layout='constrained')
    for ax,direction,label in zip(axes,DIRECTIONS,LABELS):
        values=[next(r['real_minus_gaussian_mrr'] for r in summary['paired_differences'] if r['direction']==direction and r['seed']==s) for s in [42,43,44]]
        diff=next(r for r in summary['difference_summary'] if r['direction']==direction)
        ax.axhline(0,color='#666666',lw=1)
        ax.plot([42,43,44],values,'o-',color=COLORS[0],label='Within-seed difference')
        ax.errorbar([45],[diff['mean_mrr_difference']],yerr=[diff['split_sd_mrr_difference']],fmt='s',color=COLORS[1],capsize=6,label='Mean ± sample SD')
        ax.set(title=label,xticks=[42,43,44,45],xticklabels=['42','43','44','Mean'],ylim=(-.0035,.0035))
        ax.grid(axis='y',alpha=.18)
    axes[0].set_ylabel('Real minus Gaussian MRR'); axes[1].legend(fontsize=9,loc='lower right')
    fig.suptitle('Wider grid: paired differences; SD is descriptive, not a confidence interval')
    for ext in ['png','svg']: fig.savefig(OUT/f'wider_grid_real_minus_gaussian.{ext}',dpi=200)
    plt.close(fig)
    print(f'Archived {len(runs)} wider-grid evaluations and {len(original)} separate original-grid comparisons: {OUT}')


if __name__=='__main__': main()
