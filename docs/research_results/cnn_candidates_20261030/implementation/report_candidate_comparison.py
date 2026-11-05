"""Describe the completed six-condition study; never chooses settings or fits models."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/cnn_candidate_comparison_20261030'
OUT=ROOT/'docs/research_results/cnn_candidates_20261030'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if OUT.exists(): raise FileExistsError('Preserve the prior comparison archive')
    jobs=json.loads((SOURCE/'jobs.json').read_text())
    assert len(jobs)==6 and all(j['status']=='executed' for j in jobs)
    frozen=json.loads((SOURCE/'selection_frozen.json').read_text())
    assert len(frozen['jobs'])==6 and all(j['status']=='trained' for j in frozen['jobs'])
    plan=json.loads((SOURCE/'plan.json').read_text())
    reports={}; rows=[]; contrasts=[]; within_control=[]; subject_gaps=[]
    metrics=['recall@1','recall@5','recall@10','mrr']
    for job in jobs:
        scope,control=job['scope'],job['control']; folder=SOURCE/(scope+'_'+control)
        assert sha(folder/'best.pt')==job['best_checkpoint_sha256']
        chosen=next(j for j in frozen['jobs'] if j['scope']==scope and j['control']==control)
        assert chosen['best_checkpoint_sha256']==job['best_checkpoint_sha256']
        report=json.loads((folder/'evaluation.json').read_text())
        assert report['metrics']['n_queries']==592 and len(report['candidate_keys'])==300
        assert report['candidate_keys']==plan['test_candidate_keys']
        reports[scope,control]=report
        rows.append(dict(scope=scope,control=control,best_epoch=job['best_epoch'],
                         common_validation_loss=job['validation_loss'],**{m:report['metrics'][m] for m in metrics}))
    assert len({r['candidate_pool_sha256'] for r in reports.values()})==1
    for scope in ['batch','training_unique']:
        for control in ['gaussian','shuffled']:
            real=reports[scope,'real']; other=reports[scope,control]
            contrasts.append(dict(scope=scope,comparison='real_minus_'+control,
                                   **{m:real['metrics'][m]-other['metrics'][m] for m in metrics}))
            for subject in sorted(real['subject_wise']):
                subject_gaps.append(dict(scope=scope,comparison='real_minus_'+control,subject=subject,
                    **{m:real['subject_wise'][subject][m]-other['subject_wise'][subject][m] for m in metrics}))
    for control in ['real','gaussian','shuffled']:
        before,after=reports['batch',control]['metrics'],reports['training_unique',control]['metrics']
        within_control.append(dict(control=control,**{m:after[m]-before[m] for m in metrics}))
    diff_in_diff=[]
    for control in ['gaussian','shuffled']:
        before=next(c for c in contrasts if c['scope']=='batch' and c['comparison']=='real_minus_'+control)
        after=next(c for c in contrasts if c['scope']=='training_unique' and c['comparison']=='real_minus_'+control)
        diff_in_diff.append(dict(comparison='change_in_real_minus_'+control,**{m:after[m]-before[m] for m in metrics}))
    result=dict(runs=rows,real_minus_controls=contrasts,all_training_minus_batch=within_control,
                change_in_real_minus_control=diff_in_diff,per_subject_real_minus_controls=subject_gaps,
                interpretation='Descriptive single-split controlled loss comparison. No training-accuracy success claim, no significance or replication claim; compare real EEG changes with both controls.',
                candidate_pool_sha256=next(iter(reports.values()))['candidate_pool_sha256'])
    OUT.mkdir(parents=True)
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for name,records in [('runs',rows),('real_minus_controls',contrasts),('within_control_changes',within_control),
                         ('change_in_real_minus_control',diff_in_diff),('per_subject_control_gaps',subject_gaps)]:
        with (OUT/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    ledger=[]
    for source in [SOURCE/'plan.json',SOURCE/'selection_frozen.json',SOURCE/'real_loss_validation_choice.json',SOURCE/'jobs.json',SOURCE/'summary.json',SOURCE/'retrieval.csv']:
        dest=OUT/source.name;shutil.copy2(source,dest)
        ledger.append(dict(source=str(source.relative_to(ROOT)),archived=dest.name,sha256=sha(source)))
    for folder in [SOURCE/'implementation',ROOT/'results/pairing_audit_20261030']:
        target=OUT/folder.name
        shutil.copytree(folder,target)
        for source in folder.rglob('*'):
            if source.is_file(): ledger.append(dict(source=str(source.relative_to(ROOT)),archived=str((target/source.relative_to(folder)).relative_to(OUT)),sha256=sha(source)))
    for (scope,control),report in reports.items():
        for name in ['evaluation.json','history.json','config.json','splits.json']:
            source=SOURCE/(scope+'_'+control)/name;dest=OUT/'runs'/(scope+'_'+control)/name
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
            ledger.append(dict(source=str(source.relative_to(ROOT)),archived=str(dest.relative_to(OUT)),sha256=sha(source)))
    (OUT/'archive_manifest.json').write_text(json.dumps(dict(files=ledger,checkpoint_policy='Best and last checkpoints remain in original fresh run folders; no weights copied or overwritten'),indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
    for ax,metric,scale,label,chance in zip(axes,['recall@1','mrr'],[100,1],['Recall@1 (%)','MRR'],[100/300,sum(1/i for i in range(1,301))/300]):
        for offset,control,color in [(-.23,'real','#386cb0'),(0,'gaussian','#f28e2b'),(.23,'shuffled','#59a14f')]:
            ax.bar([offset,1+offset],[reports[s,control]['metrics'][metric]*scale for s in ['batch','training_unique']],width=.22,color=color,label=control.capitalize())
        ax.axhline(chance,ls='--',color='#666666',label='Uniform ranking')
        ax.set(xticks=[0,1],xticklabels=['Batch candidates','All training texts'],ylabel=label)
        ax.spines[['right','top']].set_visible(False)
    axes[1].legend(fontsize=8)
    fig.suptitle('Fixed ZuCo1 CNN: one subject split, matched real/Gaussian/shuffled controls')
    for extension in ['png','svg']: fig.savefig(OUT/f'candidate_comparison.{extension}',dpi=200)
    plt.close(fig)
    lines=['\n## Measured held-out results\n',
           '| Loss scope | Input/labels | Selected epoch (zero-based) | Common validation loss | Recall@1 | Recall@5 | Recall@10 | MRR |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['scope']} | {row['control']} | {row['best_epoch']} | {row['common_validation_loss']:.6f} | {100*row['recall@1']:.3f}% | {100*row['recall@5']:.3f}% | {100*row['recall@10']:.3f}% | {row['mrr']:.6f} |")
    lines += ['', 'All six evaluations use the same 592 queries and 300 candidates. Uniform-ranking expectations are Recall@1 0.333%, Recall@5 1.667%, Recall@10 3.333% and MRR 0.020942. Recall values come directly from the evaluation fields, not an MRR conversion.', '',
              '![Matched CNN loss comparison](research_results/cnn_candidates_20261030/candidate_comparison.png)', '',
              '| Loss scope | Contrast | ΔRecall@1 (pp) | ΔRecall@5 (pp) | ΔRecall@10 (pp) | ΔMRR |',
              '|---|---|---:|---:|---:|---:|']
    for row in contrasts:
        lines.append(f"| {row['scope']} | {row['comparison']} | {100*row['recall@1']:+.3f} | {100*row['recall@5']:+.3f} | {100*row['recall@10']:+.3f} | {row['mrr']:+.6f} |")
    lines += ['', '| Control reference | Change in real-minus-control Recall@1 (pp) | Recall@5 (pp) | Recall@10 (pp) | MRR |',
              '|---|---:|---:|---:|---:|']
    for row in diff_in_diff:
        lines.append(f"| {row['comparison']} | {100*row['recall@1']:+.3f} | {100*row['recall@5']:+.3f} | {100*row['recall@10']:+.3f} | {row['mrr']:+.6f} |")
    lines += ['', 'The change-in-gap rows subtract the batch-arm real-minus-control gap from the all-training-arm gap. They are descriptive comparisons within this one split, not confidence intervals or independent replications. Per-subject gaps for ZAB and ZKB, full-precision tables, archived audits, configs, split manifests and checksums are in `docs/research_results/cnn_candidates_20261030/`. Baseline/new best and last checkpoints remain under the fresh run root. No training accuracy is treated as evidence of success.', '']
    doc=ROOT/'docs/cnn_candidate_comparison_20261030.md'
    old=doc.read_text(encoding='utf-8')
    if '## Measured held-out results' in old: raise FileExistsError('Results already added to report')
    doc.write_text(old+'\n'.join(lines),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
