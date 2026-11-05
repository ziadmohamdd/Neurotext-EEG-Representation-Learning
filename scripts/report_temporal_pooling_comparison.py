"""Archive/report completed validation-only pooling results; never fits a model."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from eeg_decoding.training.checkpoints import load_checkpoint
from eeg_decoding.models.temporal_pooling import matched_pooling_models, state_hash


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_pools(reports, expected_keys):
    if not expected_keys or len(expected_keys)!=len(set(expected_keys)):
        raise ValueError('Invalid planned validation pool')
    for report in reports:
        if report['evaluation_split']!='val' or report['candidate_keys']!=expected_keys:
            raise ValueError('Report must use the exact planned validation pool')
    return len(expected_keys)


def main(run):
    root=Path(__file__).resolve().parents[1]
    run=run.resolve()
    archive=root/'docs/research_results/temporal_pooling_20261030'
    doc=root/'docs/temporal_pooling_comparison_20261030.md'
    if archive.exists() or doc.exists():raise FileExistsError('Preserve previous report/archive')
    reports=json.loads((run/'validation_summary.json').read_text())
    plan=json.loads((run/'plan.json').read_text())
    decision=json.loads((run/'decision.json').read_text())
    jobs=json.loads((run/'jobs.json').read_text())
    preservation=json.loads((run/'preservation_after.json').read_text())
    assert len(reports)==6 and all(j['status']=='validation_evaluated' for j in jobs)
    assert len({(r['pooling'],r['control']) for r in reports})==6
    n_candidates=validate_pools(reports,plan['validation_candidate_keys'])
    assert all(r['metrics']['n_queries']==512 for r in reports)
    assert len({r['candidate_pool_sha256'] for r in reports})==1
    assert not decision['test_evaluated'] and not plan['punctuation_repair_applied']
    assert not list(run.glob('*/evaluation.json'))
    assert all(sha(root/k)==v for k,v in preservation['files_sha256'].items())
    assert sha(Path(plan['manifest']))==plan['manifest_sha256']
    assert sha(Path(plan['text_cache']))==plan['text_cache_sha256']
    shared_checks=[]
    old=root/'results/cnn_candidate_comparison_20261030'
    for j in jobs:
        folder=run/(j['pooling']+'_'+j['control'])
        assert sha(folder/'best.pt')==j['checkpoint_sha256']
        meta=load_checkpoint(folder/'best.pt')
        assert meta['split_digest']==plan['split_digest'] and meta['normalizer']==plan['normalizer']
        expected=plan['initialization']['global_initial_sha256' if j['pooling']=='global' else 'four_segment_initial_sha256']
        assert meta['initial_model_sha256']==expected
        if j['pooling']=='global':
            pair=matched_pooling_models(105);model=pair[0]
            load_checkpoint(folder/'best.pt',model);new_hash=state_hash(model)
            previous=load_checkpoint(old/('batch_'+j['control'])/'best.pt',model)
            old_hash=state_hash(model)
            shared_checks.append({'control':j['control'],'best_model_tensors_equal_previous_baseline':new_hash==old_hash,
                                  'best_validation_loss_equal_previous_baseline':meta['best']==previous['best']})
    archive.mkdir(parents=True)
    copies=[]
    sources=[*run.glob('*.json'),run/'validation.csv',*run.glob('*/validation.json'),*run.glob('*/history.json'),
             *run.glob('*/config.json'),*run.glob('*/splits.json'),*run.glob('implementation/*'),Path(__file__),
             root/'tests/test_pooling_report.py']
    execution_log=root/'logs/temporal_pooling_20261030.log'
    if execution_log.exists():sources.append(execution_log)
    for source in sources:
        rel=source.relative_to(run) if run in source.parents else (Path('logs') if source==execution_log else Path('implementation'))/source.name
        target=archive/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():raise FileExistsError(target)
        shutil.copyfile(source,target)
        assert sha(source)==sha(target)
        copies.append({'source':str(source.relative_to(root)),'copy':str(target.relative_to(root)),'sha256':sha(source)})
    (archive/'archive_manifest.json').write_text(json.dumps(copies,indent=2)+'\n',encoding='utf-8')
    validation={'focused_tests':'34 passed in 5.50s; tests/test_temporal_pooling.py tests/test_models.py tests/test_candidate_loss.py',
        'report_tests':'2 passed; tests/test_pooling_report.py',
        'conditions_verified':6,'validation_queries_per_condition':512,'candidate_pool_size':n_candidates,
        'same_saved_splits_and_normalizer':True,'matched_initialization_verified':True,
        'previous_artifacts_unchanged':preservation['protected_files'],'no_test_evaluation':True,
        'punctuation_repair_applied':False,'archive_copies_verified':len(copies),'baseline_reproduction_checks':shared_checks}
    (archive/'validation_checks.json').write_text(json.dumps(validation,indent=2)+'\n',encoding='utf-8')
    title='# Global-average versus four-segment CNN pooling — validation only\n\n'
    outcome=('**No useful validation improvement relative to controls. Stopped without test evaluation.**'
             if not decision['useful_validation_improvement'] else '**The descriptive validation screen passed. No test evaluation was performed; this is not evidence of generalization success.**')
    lines=[title,outcome+'\n\n',
        'This fixed, exploratory seed-42 comparison changes only temporal pooling and the input width of its linear head. '
        'All six real/Gaussian/shuffled fits used the existing export-faithful ZuCo1 targets. The proposed punctuation repair was not applied. '
        'Previous results, checkpoints and source data remain intact.\n\n',
        '## Fixed design and initialization\n\n',
        'The saved subject split is unchanged: 2,245 training observations from ZDM/ZDN/ZGW/ZJM/ZJN/ZKH/ZMG/ZPH; '
        '512 validation observations from ZJS/ZKW; the 592 test observations from ZAB/ZKB were not evaluated. '
        'The existing duration-QC cohort, 105-channel arrays at 200 Hz, training-fitted channel normalizer and deterministic '
        'center crop of at most 1,600 samples (8 seconds) are unchanged. No additional signal filtering was introduced. '
        'The existing cached 300 MiniLM sentence embeddings are reused byte-for-byte. These sentence classes recur '
        'across training and validation readers: this is held-out-reader retrieval of seen sentence classes, '
        'not unseen-sentence generalization.\n\n',
        'The loss remains symmetric multi-positive **batch** contrastive cross entropy with temperature 0.07, the original '
        'validation-selected loss. Each microbatch contains two distinct sentences; accumulation over eight microbatches '
        'does not create a 16-candidate contrastive denominator. Both architectures use AdamW (LR 0.001, weight decay 0.01), '
        'gradient clipping 1, the same seed and data/control generators, fp16-if-supported, a maximum of 50 epochs and '
        'patience 8. Each best epoch is selected solely by minimum held-out validation batch CE. Full-pool validation '
        'retrieval is computed only after that checkpoint is frozen; it never selects an epoch. Actual early-stopped '
        'durations are reported below, rather than claimed to be equal.\n\n',
        'Global averaging pools each channel to one value. The variant uses `AdaptiveAvgPool1d(4)`, concatenates the four '
        'ordered bins in channel-major order, and expands the encoder head from 32→128 to 128→128. For lengths not '
        'divisible by four, adaptive pooling can overlap boundary samples; it does not discard or pad the valid signal. '
        'These are relative temporal bins of the retained crop, not word-aligned segments. Each valid sequence is encoded '
        'separately so batch padding never enters convolution, GroupNorm or pooling.\n\n',
        '| Trainable parameters | Global | Four segments |\n|---|---:|---:|\n',
        '| Convolutional features and GroupNorm | 13,216 | 13,216 |\n',
        '| Encoder linear head including bias | 4,224 | 16,512 |\n',
        '| Text projection and LayerNorm | 50,304 | 50,304 |\n',
        f"| **Total** | **{plan['initialization']['global_parameters']:,}** | **{plan['initialization']['four_segment_parameters']:,}** |\n\n",
        'The variant adds 12,288 parameters (18.14%). Every shared tensor is initialized identically, including the '
        'convolutional stack, head bias and text projection/LayerNorm. Each of the four new head columns for a channel '
        'receives its original weight divided by four (`repeat_interleave`). This makes the initial outputs equal '
        'within numerical tolerance for equal-sized bins while allowing bin-specific gradients. For uneven bins, '
        'equal weighting of adaptive-bin means need not equal the exact global mean. The global initialization hash '
        'matches the preserved earlier CNN baseline; all three controls use the same initialization per architecture. '
        'Exact hashes and shared tensor names are in `plan.json`.\n\n',
        'Gaussian input is deterministic per observation, zero-mean in expectation and scaled by each real cropped '
        'channel’s standard deviation before the same normalizer. It retains amplitude/length structure and is not a '
        'proof of complete absence of dataset information. Shuffled control deranges training sentence classes '
        'consistently across repeated observations; its validation inputs and target keys remain real/correct. '
        'Controls and draws are identical between architectures.\n\n',
        '## Validation measurements\n\n',
        f'Recall values below are percentages of 512 validation queries against the same {n_candidates} unique sentence candidates. '
        'MRR is reported as reciprocal rank, never converted into accuracy. Epoch numbers in the table are one-based.\n\n',
        '| Pooling | Condition | Recall@1 % | Recall@5 % | Recall@10 % | MRR | Best epoch | Epochs run | Val batch CE |\n',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|\n']
    for r in reports:
        m=r['metrics'];lines.append(f"| {r['pooling']} | {r['control']} | {100*m['recall@1']:.3f} | {100*m['recall@5']:.3f} | {100*m['recall@10']:.3f} | {m['mrr']:.6f} | {r['best_epoch']+1} | {r['epochs_run']} | {r['validation_loss']:.6f} |\n")
    c=reports[0]['chance'];lines.append(f"| Uniform random ranking | analytic expectation | {100*c['recall@1']:.3f} | {100*c['recall@5']:.3f} | {100*c['recall@10']:.3f} | {c['mrr']:.6f} | — | — | — |\n\n")
    lines += ['Chance is an analytic uniform-ranking reference, not an additional trained control. Batch CE is a two-way '
              f'objective near log(2); it must not be interpreted as {n_candidates}-way retrieval accuracy.\n\n',
              '### Real-minus-control gaps\n\nRecall gaps are percentage points; MRR gaps are absolute reciprocal-rank differences.\n\n',
              '| Pooling | Comparison | ΔR@1 pp | ΔR@5 pp | ΔR@10 pp | ΔMRR |\n|---|---|---:|---:|---:|---:|\n']
    gaps=json.loads((run/'control_gaps.json').read_text())
    for g in gaps:
        lines.append(f"| {g['pooling']} | {g['comparison']} | {100*g['recall@1']:+.3f} | {100*g['recall@5']:+.3f} | {100*g['recall@10']:+.3f} | {g['mrr']:+.6f} |\n")
    lines += ['\n### Four-segment minus global changes\n\n',
              '| Condition or control gap | ΔR@1 pp | ΔR@5 pp | ΔR@10 pp | ΔMRR |\n|---|---:|---:|---:|---:|\n']
    lookup={(r['pooling'],r['control']):r for r in reports}
    architecture_changes=[]
    for condition in ('real','gaussian','shuffled'):
        delta={k:lookup['four_segment',condition]['metrics'][k]-lookup['global',condition]['metrics'][k]
               for k in ('recall@1','recall@5','recall@10','mrr')}
        architecture_changes.append(dict(comparison=condition,**delta))
    for condition in ('real_minus_gaussian','real_minus_shuffled'):
        by_gap={(g['pooling'],g['comparison']):g for g in gaps}
        delta={k:by_gap['four_segment',condition][k]-by_gap['global',condition][k]
               for k in ('recall@1','recall@5','recall@10','mrr')}
        architecture_changes.append(dict(comparison=condition,**delta))
    for d in architecture_changes:
        lines.append(f"| {d['comparison']} | {100*d['recall@1']:+.3f} | {100*d['recall@5']:+.3f} | {100*d['recall@10']:+.3f} | {d['mrr']:+.6f} |\n")
    (archive/'architecture_changes.json').write_text(json.dumps(architecture_changes,indent=2)+'\n',encoding='utf-8')
    lines += ['\n### Validation participants\n\nThese two readers form one validation split, not independent replicated cohorts.\n\n',
              '| Pooling | Condition | Participant | Queries | R@1 % | R@5 % | R@10 % | MRR |\n|---|---|---|---:|---:|---:|---:|---:|\n']
    for r in reports:
        for subject,m in r['per_subject'].items():
            lines.append(f"| {r['pooling']} | {r['control']} | {subject} | {m['n_queries']} | {100*m['recall@1']:.3f} | {100*m['recall@5']:.3f} | {100*m['recall@10']:.3f} | {m['mrr']:.6f} |\n")
    lines += ['\n## Decision and limitations\n\n',
        'Before the fits, the descriptive validation screen required higher real MRR, positive and improved real-minus-'
        'Gaussian and real-minus-shuffled MRR gaps, positive four-segment real MRR gaps for each validation participant, '
        'and at least two recall cutoffs above both controls while no worse than the global real model. A 1e-12 '
        'roundoff tolerance is used for gap improvements. This is a prespecified decision screen, not a significance '
        'threshold or a claim of clinical/scientific validity. No settings, seed, loss, epoch rule or preprocessing '
        'were changed after seeing validation retrieval.\n\n']
    for name,value in decision['checks'].items():lines.append(f"- `{name}`: {'passed' if value else 'failed'}.\n")
    lines += ['\n'+outcome+'\n\n',
        'One previously used seed/split and two validation readers cannot establish reliable EEG retrieval. Validation '
        'reuse across investigations can itself cause selection optimism. An isolated recall increase, training '
        'memorization, or a higher real score accompanied by comparable control gains is insufficient. Extra linear '
        'head capacity is confounded with temporal pooling in this comparison; the result cannot isolate a pure '
        'temporal-order mechanism. No additional seeds, hyperparameter trials, target repairs or test runs were launched.\n\n',
        'The GPU emitted the same unset `CUBLAS_WORKSPACE_CONFIG` deterministic-operation warning as the earlier '
        'experiment, plus an `adaptive_avg_pool2d_backward_cuda` nondeterministic-implementation warning for the '
        'four-bin arm. Deterministic seeding was retained with warn-only behavior, but bitwise GPU reproducibility '
        'is not assumed; this single draw does not measure run-to-run numerical variation. '
        'The archive records direct model-tensor/validation-loss reproduction checks for the three global fits.\n\n',
        '## Artifacts and validation\n\n',
        f"All old result/input files checked by the runner remained unchanged (**{preservation['protected_files']} files**). "
        'The manifest and text-cache SHA-256 values match the prior study. Model checkpoints remain only under '
        '`results/temporal_pooling_20261030/`; no prior checkpoint was overwritten. The new '
        '[archive](research_results/temporal_pooling_20261030/README.md) contains measured validation summaries, per-query '
        'ranks, participant results, candidate pools, split/normalizer/configuration provenance, history and source snapshots. '
        '`archive_manifest.json` records source/copy hashes, and `validation_checks.json` records separate artifact checks.\n\n',
        'Focused software validation: **34 passed in 5.50 s**, no skips, for `tests/test_temporal_pooling.py`, '
        '`tests/test_models.py` and `tests/test_candidate_loss.py`. Checks cover baseline compatibility, matched '
        'initialization, parameter counts, padding/short sequences, segment ordering/gradients, unchanged loss and '
        'control-relative screening. Two additional report bookkeeping tests passed for exact validation-pool '
        'matching and rejection of test/different candidate pools. These checks caught and corrected an initial '
        'report-only assumption of 300 validation candidates: the measured evaluation pool has 298; the training '
        'bank/cache has 300. All six evaluations used 298 from the outset, so no ranks or model outputs were changed. '
        'This is not a rerun of the previously recorded full suite.\n\n',
        'Commands executed with the existing environment:\n\n```powershell\n'
        '.\\.venv\\Scripts\\python.exe -m pytest tests/test_temporal_pooling.py tests/test_models.py tests/test_candidate_loss.py -q\n'
        '.\\.venv\\Scripts\\python.exe -m pytest tests/test_pooling_report.py -q\n'
        '.\\.venv\\Scripts\\python.exe -u scripts/run_temporal_pooling_comparison.py --output results/temporal_pooling_20261030\n'
        '.\\.venv\\Scripts\\python.exe scripts/report_temporal_pooling_comparison.py --run results/temporal_pooling_20261030\n```\n'
        '\nRunner and reporter refuse existing output destinations. To reproduce, use fresh versioned paths.\n']
    doc.write_text(''.join(lines),encoding='utf-8')
    (archive/'README.md').write_text('# Validation-only temporal pooling comparison\n\n'
        'See the [full report](../../temporal_pooling_comparison_20261030.md).\n\n'+outcome+'\n\n'
        '`plan.json` fixes the recipe, initialization, validation decision rule and test prohibition before fitting. '
        '`validation_summary.json`, `validation.csv`, condition-level `validation.json` and `control_gaps.json` '
        'contain only validation measurements. `decision.json` records the screen. Full checkpoints remain in the '
        'new results directory. Original targets, including the proposed punctuation repairs, were unchanged.\n\n'
        'The variant has 80,032 parameters versus 67,744, with exact shared-layer initialization. All six fits used '
        'the original batch contrastive loss, saved seed-42 splits and common training budget. '
        'No test retrieval was performed. `validation_checks.json` and `archive_manifest.json` preserve checksums '
        'and separate verification outcomes.\n',encoding='utf-8')
    print(json.dumps({'report':str(doc),'validation':validation,'decision':decision},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    main(parser.parse_args().run)
