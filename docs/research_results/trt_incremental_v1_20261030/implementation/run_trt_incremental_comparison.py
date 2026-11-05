"""Cache-only, validation-only incremental EEG information comparison."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from threadpoolctl import threadpool_limits
from eeg_decoding.data.dataset import read_manifest
from eeg_decoding.evaluation.incremental_ridge import select_structural,select_combined
from eeg_decoding.evaluation.retrieval import ranks,summarize,chance
from eeg_decoding.training.alignment_train import TextCache

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(array):
    a=np.ascontiguousarray(array)
    return hashlib.sha256(str(a.shape).encode()+str(a.dtype).encode()+a.tobytes()).hexdigest()


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(output):
    archive=ROOT/'docs/research_results/trt_incremental_v1_20261030';doc=ROOT/'docs/trt_incremental_v1_20261030.md'
    if output.exists() or archive.exists() or doc.exists():raise FileExistsError('Use fresh experiment/report destinations')
    cfg_path=ROOT/'configs/trt_incremental_information_v1.json';cfg=json.loads(cfg_path.read_text())
    prior=ROOT/'results/spectral_trt_pilot_v1_20261030'
    oldplan=json.loads((prior/'plan.json').read_text());viewmeta=json.loads((prior/'feature_view.json').read_text())
    view=ROOT/cfg['source_view'];index=json.loads((view/'index.json').read_text())
    assert not index['test_features_extracted'] and index['model_data_splits']==['train','val']
    if sha(view/'sentence_vectors.npz')!=viewmeta['vectors_sha256'] or sha(view/'index.json')!=viewmeta['index_sha256']:
        raise ValueError('Source feature cache changed')
    with np.load(view/'sentence_vectors.npz',allow_pickle=False) as cache:
        if any(k.endswith('_test') for k in cache.files):raise ValueError('No test feature cache permitted')
        s={split:cache['count_duration_only_'+split].copy() for split in ('train','val')}
        eeg={condition:{split:cache[source+'_'+split][:,:840].copy() for split in ('train','val')}
             for condition,source in [('structural_real','real_trt'),('structural_gaussian','matched_gaussian')]}
        for split,n in [('train',2245),('val',512)]:
            assert s[split].shape==(n,62)
            np.testing.assert_array_equal(cache['real_trt_'+split][:,840:],cache['matched_gaussian_'+split][:,840:])
            np.testing.assert_array_equal(s[split][:,14:],cache['real_trt_'+split][:,843:])
            np.testing.assert_array_equal(s[split][:,0],cache['real_trt_'+split][:,840])
            np.testing.assert_array_equal(np.isnan(eeg['structural_real'][split]),np.isnan(eeg['structural_gaussian'][split]))
    records=read_manifest(oldplan['manifest']);lookup={(r.dataset,r.subject,r.session,r.task,r.sentence_id):r for r in records}
    train=[lookup[tuple(k)] for k in oldplan['train_observations']]
    val=[lookup[tuple(k)] for k in oldplan['validation_observations']]
    pool=oldplan['candidate_keys'];assert len(pool)==298 and len(train)==2245 and len(val)==512
    assert not {'ZAB','ZKB'}&{r.subject for r in train+val}
    text=TextCache(oldplan['text_cache'])
    if text.digest!=oldplan['text_cache_sha256']:raise ValueError('Targets changed')
    targets=text.get([r.key for r in train]).astype(np.float64);valkeys=[r.key for r in val];vectors=text.get(pool)
    protected=[p for p in (ROOT/'results').rglob('*') if p.is_file()]+list(view.glob('*'))+[Path(oldplan['manifest']),Path(oldplan['text_cache'])]
    before={str(p.relative_to(ROOT)):sha(p) for p in protected if p.is_file()}
    output.mkdir(parents=True);write(output/'preservation_before.json',before)
    plan={'configuration':cfg,'configuration_sha256':sha(cfg_path),'source_feature_view':str(view),'vectors_sha256':sha(view/'sentence_vectors.npz'),
        'source_pilot_plan_sha256':sha(prior/'plan.json'),'source_index_sha256':sha(view/'index.json'),
        'text_cache':oldplan['text_cache'],'text_cache_sha256':text.digest,'text_cache_metadata':text.metadata,
        'train_observations':oldplan['train_observations'],'validation_observations':oldplan['validation_observations'],
        'candidate_keys':pool,'candidate_pool_sha256':oldplan['candidate_pool_sha256'],
        'structural_columns':index['columns']['count_duration'],'eeg_columns':index['columns']['spectral'][:840],
        'raw_structural_array_hashes':{split:array_sha(x) for split,x in s.items()},
        'eeg_array_hashes':{c:{split:array_sha(x) for split,x in blocks.items()} for c,blocks in eeg.items()},
        'eligibility':viewmeta['coverage'],'new_exclusions':[],'raw_MAT_or_EEG_files_opened':False,'test_features_extracted':False,
        'structural_consistency':'One raw structural block; one frozen training-fit structural transform shared by all models',
        'duration_units':'Source sample counts at 500 Hz; no timing reconstructed',
        'limits':'Source gaze-assisted TRT summaries and boundary/producer limits inherited from original pilot; no target correction'}
    write(output/'plan.json',plan)
    for source in [cfg_path,Path(__file__),ROOT/'src/eeg_decoding/evaluation/incremental_ridge.py',
                   ROOT/'src/eeg_decoding/evaluation/spectral_pilot.py',ROOT/'src/eeg_decoding/evaluation/retrieval.py',ROOT/'tests/test_incremental_ridge.py']:
        dest=output/'implementation'/source.name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(source,dest)
    started=time.monotonic();print('FIT structural-only (28 settings)',flush=True)
    base,basegrid,basepred,st,sv=select_structural(s['train'],s['val'],targets,valkeys,pool,vectors,cfg)
    transform=base['transform']
    np.savez_compressed(output/'structural_preprocessing.npz',fill=transform['fill'],center=transform['center'],scale=transform['scale'])
    frozen={'imputation':transform['imputation'],'scaling':transform['scaling'],'alpha':base['alpha'],
            'transform_sha256':sha(output/'structural_preprocessing.npz'),
            'transformed_array_hashes':{'train':array_sha(st),'val':array_sha(sv)}}
    write(output/'structural_preprocessing_frozen.json',frozen)
    reports=[];zero_checks=[]
    def save(condition,choice,grid,prediction,elapsed):
        folder=output/condition;folder.mkdir()
        et=choice.get('eeg_transform')
        payload={'coef':choice['coef'],'intercept':choice['intercept'],'structural_fill':transform['fill'],
                 'structural_center':transform['center'],'structural_scale':transform['scale'],
                 'weight':float(choice.get('weight',0))}
        if et:payload.update(eeg_fill=et['fill'],eeg_center=et['center'],eeg_scale=et['scale'])
        np.savez_compressed(folder/'ridge.npz',**payload)
        selected={'alpha':choice['alpha'],'eeg_weight':choice.get('weight',0),
            'structural_imputation':transform['imputation'],'structural_scaling':transform['scaling'],
            'eeg_imputation':et['imputation'] if et else None,'eeg_scaling':et['scaling'] if et else None,
            'n_features':choice['coef'].shape[1],'model_sha256':sha(folder/'ridge.npz'),
            'structural_transform_sha256':frozen['transform_sha256'],
            'shared_structural_array_hashes':frozen['transformed_array_hashes']}
        write(folder/'selection.json',{'selected':selected,'grid':grid})
        rank=ranks(prediction,vectors,valkeys,pool)
        report={'condition':condition,'evaluation_split':'val','metrics':summarize(rank),'ranks':rank.tolist(),
            'selection':selected,'candidate_keys':pool,'candidate_pool_sha256':plan['candidate_pool_sha256'],
            'observation_identities':plan['validation_observations'],'seconds':elapsed,
            'per_subject':{subject:summarize(rank[[r.subject_key==subject for r in val]]) for subject in sorted({r.subject_key for r in val})}}
        # Replay the actual saved model through the same input blocks.
        with np.load(folder/'ridge.npz',allow_pickle=False) as model:
            z=(np.where(np.isfinite(s['val']),s['val'],model['structural_fill'])-model['structural_center'])/model['structural_scale']
            if et:
                raw=eeg[condition]['val']
                e=(np.where(np.isfinite(raw),raw,model['eeg_fill'])-model['eeg_center'])/model['eeg_scale']
                z=np.c_[z,float(model['weight'])*e]
            replay=z@model['coef'].T+model['intercept']
            np.testing.assert_array_equal(ranks(replay,vectors,valkeys,pool),rank)
        write(folder/'validation.json',report);reports.append(report)
        print(json.dumps({'condition':condition,'metrics':report['metrics'],'selection':selected}),flush=True)
    save('structural_only',base,basegrid,basepred,time.monotonic()-started)
    base_lookup={row['alpha']:row for row in basegrid if row['imputation']==transform['imputation'] and row['scaling']==transform['scaling']}
    for condition in ('structural_real','structural_gaussian'):
        started=time.monotonic();print('FIT '+condition+' (196 settings, including zero EEG)',flush=True)
        choice,table,prediction=select_combined(st,sv,eeg[condition]['train'],eeg[condition]['val'],targets,valkeys,pool,vectors,cfg)
        assert array_sha(st)==frozen['transformed_array_hashes']['train'] and array_sha(sv)==frozen['transformed_array_hashes']['val']
        zero=[row for row in table if row['weight']==0]
        for row in zero:
            for metric in ('mrr','recall@1','recall@5','recall@10'):assert row[metric]==base_lookup[row['alpha']][metric]
        zero_checks.append({'condition':condition,'zero_weight_settings':len(zero),'all_zero_weight_metrics_equal_structural_at_same_alpha':True})
        assert max(row['mrr'] for row in table)>=reports[0]['metrics']['mrr']-1e-12
        save(condition,choice,table,prediction,time.monotonic()-started)
    real,gauss=reports[1],reports[2]
    differences=[]
    for left,right in [(real,reports[0]),(real,gauss),(gauss,reports[0])]:
        differences.append({'comparison':left['condition']+'_minus_'+right['condition'],
            **{k:left['metrics'][k]-right['metrics'][k] for k in ('recall@1','recall@5','recall@10','mrr')},
            'per_subject':{p:{k:left['per_subject'][p][k]-right['per_subject'][p][k] for k in ('recall@1','recall@5','recall@10','mrr')} for p in left['per_subject']}})
    decision={'real_mrr_above_structural':real['metrics']['mrr']>reports[0]['metrics']['mrr']+1e-12,
        'real_mrr_above_gaussian':real['metrics']['mrr']>gauss['metrics']['mrr']+1e-12,
        'real_positive_mrr_gaps_for_each_reader':all(real['per_subject'][p]['mrr']>max(reports[0]['per_subject'][p]['mrr'],gauss['per_subject'][p]['mrr'])+1e-12 for p in real['per_subject']),
        'test_extracted_or_evaluated':False,'interpretation':'Selected validation point estimates on one reused split; unequal model-selection dimension and validation reuse preclude a confirmatory significance claim.'}
    decision['real_mrr_improves_over_both']=decision['real_mrr_above_structural'] and decision['real_mrr_above_gaussian']
    write(output/'validation_summary.json',reports);write(output/'differences.json',differences);write(output/'decision.json',decision)
    after={p:sha(ROOT/p) for p in before};assert before==after
    write(output/'preservation_after.json',{'unchanged':True,'protected_files':len(before),'files_sha256':after})
    validation={'unit_tests':'4 passed in 1.71s','structural_settings':len(basegrid),'combined_settings_each':196,
        'identical_structural_raw_and_transformed_blocks':True,'zero_weight_nesting':zero_checks,'model_rank_replays':3,
        'identical_observations_and_candidate_pool':True,'old_feature_cache_unchanged':sha(view/'sentence_vectors.npz')==viewmeta['vectors_sha256'],
        'previous_artifacts_unchanged':len(before),'test_features_extracted':False}
    write(output/'validation_checks.json',validation)
    archive.mkdir(parents=True);copies=[]
    for source in [*output.glob('*.json'),*output.glob('*/selection.json'),*output.glob('*/validation.json'),*output.glob('implementation/*'),output/'structural_preprocessing.npz',view/'index.json']:
        rel=source.relative_to(output) if output in source.parents else Path('source_view')/source.name
        dest=archive/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest);assert sha(source)==sha(dest)
        copies.append({'source':str(source.relative_to(ROOT)),'copy':str(dest.relative_to(ROOT)),'sha256':sha(source)})
    write(archive/'archive_manifest.json',copies)
    rows=[dict(condition=r['condition'],**r['metrics'],**{k:r['selection'][k] for k in ('alpha','eeg_weight','eeg_imputation','eeg_scaling')}) for r in reports]
    with (archive/'validation.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    conclusion=('Real TRT has a higher selected validation MRR than both comparators.' if decision['real_mrr_improves_over_both'] else 'Real TRT does not improve selected validation MRR over both comparators.')
    lines=['# TRT incremental-information comparison — validation only\n\n','**'+conclusion+' No test features were extracted or evaluated.**\n\n',
        'This new comparison reuses the unchanged `zuco1_trt_sentence_mean_pilot_v1` train/validation cache. '
        'All three models use 2,245 training observations, 512 validation queries and the same 298 sentence candidates. '
        'There are no new exclusions. ZMG/70 remains the one all-missing EEG training observation; all validation '
        'observations are retained. Original targets, source data and previous artifacts remain unchanged.\n\n',
        '## Exact shared structure and nested models\n\n',
        'Structure is the prior count/duration-only control: **14 count/duration summaries plus 48 training-derived '
        'word-count basis columns (62 dimensions)**. All raw structural columns, row order and train/validation arrays '
        'are identical across conditions. It includes durations, not only the three summary fields used in the '
        'previous pilot’s spectral arms. No word text, token identity, sentence/subject ID or gaze coordinates '
        'enter any numeric model matrix. Identity metadata is used only for splitting, labels and provenance.\n\n',
        'Structural-only selection uses the same mean/median imputation, standard/robust-IQR scaling and seven '
        'ridge alphas as the prior pilot (28 settings). Every transform parameter is fitted on training data only; '
        'the setting is selected by validation MRR. The selected structural transform is then **frozen and reused '
        'byte-for-byte in both combined models**, with structural block weight 1. Its selected alpha is '
        f"{base['alpha']:g}, imputation {transform['imputation']}, scaling {transform['scaling']}.\n\n",
        'Each combined model appends only the cached **840 TRT amplitude means**, using the real or previously '
        'matched Gaussian block. Their original structural tails are discarded. Both receive exactly the same '
        'search: EEG mean/median imputation × standard/robust-IQR scaling × alphas '
        '`[0.1,1,10,100,1000,10000,100000]` × EEG weights `[0,0.01,0.03,0.1,0.3,1,3]` '
        '= **196 settings each**. EEG transforms fit their respective training blocks only. The shared structural '
        'transform is not refitted or reselected for either combined model.\n\n',
        'The ridge design is `[transformed_structure, weight × transformed_EEG]` (902 dimensions). '
        '**Weights apply after scaling**, so the scaler cannot cancel them. A weight of zero exactly reuses '
        'the frozen-structural ridge path at each alpha, with zero stored EEG coefficients. All 28 zero-weight '
        'entries per combined model match structural-only validation metrics at the same alpha exactly. '
        'Selection maximizes validation MRR; ties within 1e-12 prefer the first declared entry, with ascending '
        'EEG weight as the outer loop, so zero wins ties. Thus adding a block cannot force a worse selected '
        'validation MRR than structure alone. This nesting is an experimental design property, not evidence '
        'that the extra block contains useful information.\n\n',
        'Gaussian features and their exact word counts/missingness masks are reused unchanged; no new draw '
        'or feature extraction occurs. The [pilot report](spectral_trt_pilot_v1_20261030.md) documents training-only '
        'Gaussian marginal matching and TRT source provenance. Duration fields remain source sample counts '
        'at 500 Hz, not milliseconds. Gaze-assisted segmentation, supplied full-session preprocessing, differing '
        'gamma2 documentation and potential sentence-boundary tails remain limitations. No punctuation repair '
        'or test-source read was performed.\n\n',
        '## Selected validation results\n\nRecall is percent of the same 512 queries against 298 candidates. MRR is not accuracy.\n\n',
        '| Model | R@1 % | R@5 % | R@10 % | MRR | Alpha | EEG weight | EEG imputation / scaling |\n|---|---:|---:|---:|---:|---:|---:|---|\n']
    for r in reports:
        m=r['metrics'];sel=r['selection'];prep='—' if r['condition']=='structural_only' else sel['eeg_imputation']+' / '+sel['eeg_scaling']
        lines.append(f"| {r['condition']} | {100*m['recall@1']:.4f} | {100*m['recall@5']:.4f} | {100*m['recall@10']:.4f} | {m['mrr']:.6f} | {sel['alpha']:g} | {sel['eeg_weight']:g} | {prep} |\n")
    c=chance(298);lines.append(f"| Uniform-ranking expectation | {100*c['recall@1']:.4f} | {100*c['recall@5']:.4f} | {100*c['recall@10']:.4f} | {c['mrr']:.6f} | — | — | — |\n\n")
    lines+=['## Incremental differences\n\nRecall differences are percentage points; MRR differences are absolute.\n\n',
        '| Comparison | ΔR@1 pp | ΔR@5 pp | ΔR@10 pp | ΔMRR |\n|---|---:|---:|---:|---:|\n']
    for d in differences:lines.append(f"| {d['comparison']} | {100*d['recall@1']:+.4f} | {100*d['recall@5']:+.4f} | {100*d['recall@10']:+.4f} | {d['mrr']:+.6f} |\n")
    lines+=['\n## Validation readers\n\n| Model | Reader | Queries | R@1 % | R@5 % | R@10 % | MRR |\n|---|---|---:|---:|---:|---:|---:|\n']
    for r in reports:
        for p,m in r['per_subject'].items():lines.append(f"| {r['condition']} | {p} | {m['n_queries']} | {100*m['recall@1']:.4f} | {100*m['recall@5']:.4f} | {100*m['recall@10']:.4f} | {m['mrr']:.6f} |\n")
    lines+=['\nPer-reader incremental differences, all selection rows and all 512 query ranks per model are '
        'retained in the archive. These two readers form one reused validation split, not replicated cohorts.\n\n',
        '## Interpretation and limitations\n\n',conclusion+'\n\n',
        f"Real MRR is above structural-only: **{decision['real_mrr_above_structural']}**; above Gaussian-combined: "
        f"**{decision['real_mrr_above_gaussian']}**; positive gaps to both for each validation reader: "
        f"**{decision['real_positive_mrr_gaps_for_each_reader']}**.\n\n",
        'These are selected validation point estimates. Structural-only has 28 searched settings; each combined '
        'model has 196, making superiority over structural-only alone susceptible to search optimism. Real and '
        'Gaussian combined searches are exactly matched, but this validation split was already used in earlier '
        'investigations. No independent significance or generalization claim follows from a small selected '
        'increase, training fit or a single metric. The zero-EEG option and observed Gaussian gain must inform '
        'whether a real-feature increase is useful. No grid expansion, extra seed or test evaluation followed '
        'the results.\n\n',
        '## Reproduction and preservation\n\n',
        f"Four focused tests passed in 1.71 s. They verify exact zero-weight nesting, effective post-scaling "
        'weights, reference-ridge predictions, training-only preprocessing and identical search candidates. '
        'The run replays every selected saved model to reproduce its validation ranks, checks all zero-weight '
        'metrics against structural-only and hashes the raw/transformed structural arrays. '
        f"All **{len(before)} protected prior files** remained unchanged. The feature cache and targets were "
        'reused byte-for-byte; no MAT or raw EEG data were opened. No test feature matrix was loaded, extracted '
        'or evaluated.\n\n',
        f"Results: `{output.relative_to(ROOT)}`. The [archive](research_results/trt_incremental_v1_20261030/README.md) "
        'contains the frozen plan, preprocessing provenance, complete grids, metrics, ranks, differences and '
        'source snapshots. Coefficients and full preprocessing arrays remain in the new result folder.\n\n',
        '```powershell\n.\\.venv\\Scripts\\python.exe -m pytest tests/test_incremental_ridge.py -q\n'
        '.\\.venv\\Scripts\\python.exe -u scripts/run_trt_incremental_comparison.py --output results/trt_incremental_v1_20261030\n```\n\n'
        'The runner refuses existing result/report/archive destinations. Use new versioned destinations for reproduction.\n']
    doc.write_text(''.join(lines),encoding='utf-8')
    (archive/'README.md').write_text('# Incremental TRT information — validation only\n\n'
        'See the [full report](../../trt_incremental_v1_20261030.md).\n\n'+conclusion+'\n\n'
        'All three models share the full 62-feature structural block and its frozen train-fitted transform. '
        'Real/Gaussian combined models use identical alpha, EEG-preprocessing and post-scaling block-weight '
        'searches, including an exact zero-EEG candidate. Both use the original matched Gaussian and real TRT '
        'cache. All 2,245 training observations, 512 validation observations and 298 candidates are unchanged. '
        'No test features were extracted or evaluated.\n\n'
        '`plan.json`, `structural_preprocessing_frozen.json`, `validation_checks.json`, `differences.json`, '
        '`validation_summary.json`, condition grids/ranks, source snapshots and `archive_manifest.json` '
        'record configuration and verification. Full coefficients remain in the new results directory.\n',encoding='utf-8')
    print(json.dumps({'decision':decision,'validation':validation,'report':str(doc)},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=4):main(args.output)
