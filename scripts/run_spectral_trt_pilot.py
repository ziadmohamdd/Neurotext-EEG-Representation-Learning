"""Versioned ZuCo1 TRT ridge pilot. Train/validation only, no test code path."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from threadpoolctl import threadpool_limits
from eeg_decoding.data.dataset import read_manifest, sentence_key, observation_seed, load_signal
from eeg_decoding.data.trt_view import iter_trt_sentences,mean_features,structure,count_duration,BANDS
from eeg_decoding.data.validation import manifest_digest
from eeg_decoding.evaluation.controls import shuffled_keys
from eeg_decoding.evaluation.retrieval import ranks,summarize,chance
from eeg_decoding.evaluation.spectral_pilot import fit_word_moments,matched_gaussian,count_basis,select_ridge,useful_validation
from eeg_decoding.models.classical import eeg_features
from eeg_decoding.preprocessing.normalization import ChannelNormalizer
from eeg_decoding.training.alignment_train import TextCache

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def identity(r):return (r.dataset,r.subject,r.session,r.task,r.sentence_id)


def main(output,view):
    if output.exists() or view.exists():raise FileExistsError('Use fresh result and feature-view directories')
    cfg_path=ROOT/'configs/spectral_trt_pilot_v1.json';cfg=json.loads(cfg_path.read_text())
    manifest=ROOT/'data/manifests/zuco1_duration_qc.jsonl'
    prior=json.loads((ROOT/'results/cnn_candidate_comparison_20261030/plan.json').read_text())
    text_path=Path(prior['text_cache'])
    if sha(manifest)!=prior['manifest_sha256'] or sha(text_path)!=prior['text_cache_sha256']:raise ValueError('Original targets/cohort changed')
    saved_path=ROOT/'results/cnn_candidate_comparison_20261030/batch_real/splits.json'
    saved=json.loads(saved_path.read_text());records=read_manifest(manifest)
    by_identity={identity(r):r for r in records}
    splits={name:[by_identity[tuple(x)] for x in saved['splits'][name]['observations']] for name in ('train','val')}
    if len(splits['train'])!=2245 or len(splits['val'])!=512:raise ValueError('Unexpected cohort')
    test_subjects=set(saved['splits']['test']['subjects'])
    assert not test_subjects & {r.subject_key for rows in splits.values() for r in rows}
    for name,rows in splits.items():
        if manifest_digest(rows)!=saved['splits'][name]['digest']:raise ValueError('Original array/split changed')
    pool=saved['splits']['val']['sentence_keys']
    assert len(pool)==298 and pool==sorted({r.key for r in splits['val']})
    text=TextCache(text_path)
    protected=[p for p in (ROOT/'results').rglob('*') if p.is_file()]+[manifest,text_path]
    protected += list((ROOT/'data/processed').glob('zuco*_batch/combined.jsonl'))
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    output.mkdir(parents=True);view.mkdir(parents=True)
    write(output/'preservation_before.json',before)
    columns=[f'TRT_{b}:{channel}' for b in BANDS for channel in records[0].channels]
    plan={'configuration':cfg,'configuration_sha256':sha(cfg_path),'manifest':str(manifest),'manifest_sha256':sha(manifest),
        'text_cache':str(text_path),'text_cache_sha256':sha(text_path),'saved_split_source':str(saved_path),'saved_split_sha256':sha(saved_path),
        'train_observations':[list(identity(r)) for r in splits['train']],'validation_observations':[list(identity(r)) for r in splits['val']],
        'candidate_keys':pool,'candidate_pool_sha256':hashlib.sha256(json.dumps(pool).encode()).hexdigest(),
        'source_field_order':columns,'duration_units':'sample counts at source 500 Hz; no onset inferred',
        'aggregation':'Per-coordinate finite word mean, equally weighted across words; supplied TRT itself averages available fixation envelope means',
        'source_code':{'path':'results/pairing_audit_20261030/official_firstLevel_NR.m','sha256':sha(ROOT/'results/pairing_audit_20261030/official_firstLevel_NR.m')},
        'provenance_limitations':['Gaze-assisted word assignment and structural features are privileged annotations',
            'Producer gamma2 code 40.5–49.5 Hz versus descriptor 40–49.5 Hz; exact producer revision not proved for all exports',
            'Provided word features use source fixation windows, not the sentence baseline eight-second crop; sampled boundary tails exist',
            'Supplied session-level preprocessing cannot be retroactively made train-fitted; all new learned transforms use training only'],
        'word_text_or_sentence_identity_as_input':False,'new_exclusions':[],
        'test_policy':'No test input extraction or evaluation in this runner; validation screen recorded before any later decision'}
    write(output/'plan.json',plan)
    for source in [cfg_path,Path(__file__),ROOT/'src/eeg_decoding/data/trt_view.py',ROOT/'src/eeg_decoding/data/mat5.py',
                   ROOT/'src/eeg_decoding/evaluation/spectral_pilot.py',ROOT/'src/eeg_decoding/models/classical.py',
                   ROOT/'src/eeg_decoding/evaluation/controls.py',ROOT/'src/eeg_decoding/evaluation/retrieval.py',ROOT/'tests/test_spectral_pilot.py']:
        destination=output/'implementation'/source.name;destination.parent.mkdir(exist_ok=True);shutil.copyfile(source,destination)
    wanted=defaultdict(dict)
    for name,rows in splits.items():
        for r in rows:wanted[r.subject][int(r.sentence_id)]=r
    source_plan=json.loads((ROOT/'data/raw/zuco1/download_plan.json').read_text())
    expected={p['name']:p['sha256'] for p in source_plan['files'] if p['role']=='sentences'}
    data={};sources=[]
    for subject,observations in sorted(wanted.items()):
        source=ROOT/f'data/raw/zuco1/results{subject}_NR.mat'
        print('EXTRACT TRT only '+subject,flush=True)
        digest=sha(source)
        if digest!=expected[source.name]:raise ValueError('Source checksum changed')
        found=set()
        for index,parent_text,words,durations,fixations in iter_trt_sentences(source):
            if index not in observations:continue
            r=observations[index]
            if sentence_key(parent_text)!=r.key or parent_text!=r.text:raise ValueError('Source/target pairing changed')
            if words.shape[1]!=840:raise ValueError('Feature width mismatch')
            # Current corpus masks are jointly missing by word. No text or IDs
            # enter the matrix; identities only bind storage, labels and RNG.
            valid=np.isfinite(words)
            if not np.all(valid.all(1)|~valid.any(1)):raise ValueError('Unexpected partial-word mask; review rather than silently change protocol')
            data[identity(r)]=(words,durations,fixations)
            found.add(index)
        if found!=set(observations):raise ValueError('Missing eligible observation')
        sources.append({'subject':subject,'source':str(source.resolve()),'sha256':digest,'eligible_observations':len(found)})
        print(f'{subject}: {len(found)} eligible observations paired',flush=True)
    assert len(data)==2757
    moments=fit_word_moments(data[identity(r)][0] for r in splits['train'])
    matrices={};coverage=[];mask_proofs=[]
    for name,rows in splits.items():
        eeg=[];noise=[];mask=[];counts=[]
        for r in rows:
            words,durations,fixations=data[identity(r)];stat=structure(words)
            gaussian=matched_gaussian(words,moments,observation_seed(r,cfg['seed']))
            core=mean_features(words)
            eeg.append(np.r_[core,stat]);noise.append(np.r_[mean_features(gaussian),stat])
            mask.append(np.r_[np.where(np.isfinite(core),0.,np.nan),stat]);counts.append(count_duration(words,durations,fixations))
            proof=hashlib.sha256(np.packbits(np.isfinite(words)).tobytes()).hexdigest()
            assert proof==hashlib.sha256(np.packbits(np.isfinite(gaussian)).tobytes()).hexdigest()
            mask_proofs.append({'identity':list(identity(r)),'split':name,'words':len(words),'valid_words':int(stat[1]),
                               'shape':list(words.shape),'real_mask_sha256':proof,'gaussian_mask_sha256':proof})
        matrices[name]={'real_trt':np.asarray(eeg),'matched_gaussian':np.asarray(noise),
                        'zero_missingness_only':np.asarray(mask),'count_duration_only':np.asarray(counts)}
        coverage.append({'split':name,'observations':len(rows),'word_observations':sum(len(data[identity(r)][0]) for r in rows),
            'complete_word_observations':int(sum(structure(data[identity(r)][0])[1] for r in rows)),
            'all_missing_observations':[list(identity(r)) for r in rows if not np.isfinite(data[identity(r)][0]).any()],
            'excluded_observations':[]})
    training_word_counts=matrices['train']['real_trt'][:,840].copy()
    for name in ('train','val'):
        basis,levels=count_basis(training_word_counts,matrices[name]['real_trt'][:,840])
        for condition in matrices[name]:matrices[name][condition]=np.c_[matrices[name][condition],basis]
        matrices[name]['class_shuffled']=matrices[name]['real_trt']
    write(view/'index.json',{'version':cfg['version'],'sources':sources,'coverage':coverage,'mask_proofs':mask_proofs,
        'columns':{'spectral':columns+cfg['shared_structure']+[f'word_count_equals_{v}' for v in levels]+['word_count_unseen'],
                   'count_duration':['word_count','fixated_word_count','complete_word_count','missing_word_count','missing_fraction',
                       'fixation_total','fixations_mean_over_nonempty_words','fixations_sd_over_nonempty_words','TRT_total_samples',
                       'TRT_mean_samples','TRT_sd_samples','TRT_min_samples','TRT_median_samples','TRT_max_samples']+[f'word_count_equals_{v}' for v in levels]+['word_count_unseen']},
        'word_count_basis_training_levels':levels.tolist(),'identity_use':'Index/labels/RNG only; never model-input columns',
        'word_text_in_cache':False,'model_data_splits':['train','val'],'test_features_extracted':False})
    np.savez_compressed(view/'gaussian_training_moments.npz',**moments)
    del data
    # Refit the existing 1050-feature sentence ridge recipe on these same rows.
    old=json.loads((ROOT/'results/temporal_pooling_20261030/plan.json').read_text())
    if old['splits']!=saved:raise ValueError('Sentence baseline normalizer split differs')
    assert set(old['normalizer']['subjects'])=={r.subject_key for r in splits['train']}
    norm=ChannelNormalizer.from_state(old['normalizer'])
    write(view/'sentence_baseline_normalizer.json',{'source':'results/temporal_pooling_20261030/plan.json','normalizer':old['normalizer'],
                                                  'recipe':'Existing train-fitted voltage moments, 200 Hz arrays, center crop max1600, eeg_features() unchanged'})
    for name,rows in splits.items():
        print('EXTRACT existing sentence features '+name,flush=True)
        features=[]
        for r in rows:
            x=load_signal(r)
            if x.shape[1]>1600:
                start=(x.shape[1]-1600)//2;x=x[:,start:start+1600]
            features.append(eeg_features(norm.transform(x),r.sfreq))
        matrices[name]['existing_sentence_ridge']=np.asarray(features,dtype=np.float64)
    payload={condition+'_'+name:x for name,by in matrices.items() for condition,x in by.items() if condition!='class_shuffled'}
    np.savez_compressed(view/'sentence_vectors.npz',**payload)
    write(output/'feature_view.json',{'path':str(view.resolve()),'version':cfg['version'],'index_sha256':sha(view/'index.json'),
        'vectors_sha256':sha(view/'sentence_vectors.npz'),'gaussian_training_moments_sha256':sha(view/'gaussian_training_moments.npz'),
        'aliases':{'class_shuffled':'real_trt'},'source_hashes':sources,'coverage':coverage})
    jobs=[];reports=[]
    original=[r.key for r in splits['train']];shuffled=shuffled_keys(original,cfg['seed'])
    write(output/'class_shuffle_mapping.json',dict(sorted(set(zip(original,shuffled)))))
    for condition in cfg['conditions']:
        started=time.monotonic();print('FIT validation-only '+condition,flush=True)
        keys=shuffled if condition=='class_shuffled' else original
        selected,grid,prediction=select_ridge(matrices['train'][condition],matrices['val'][condition],text.get(keys).astype(np.float64),
            [r.key for r in splits['val']],pool,text.get(pool),cfg)
        folder=output/condition;folder.mkdir()
        transform=selected['transform']
        np.savez_compressed(folder/'ridge.npz',coef=selected['coef'],intercept=selected['intercept'],
            imputer_fill=transform['fill'],scaler_center=transform['center'],scaler_scale=transform['scale'])
        selection={'alpha':selected['alpha'],'imputation':transform['imputation'],'scaling':transform['scaling'],
                   'all_missing_training_columns':transform['all_missing_training_columns'],'grid':grid,
                   'n_features':matrices['train'][condition].shape[1],'coefficient_sha256':sha(folder/'ridge.npz')}
        write(folder/'selection.json',selection)
        rank=ranks(prediction,text.get(pool),[r.key for r in splits['val']],pool)
        report={'condition':condition,'evaluation_split':'val','metrics':summarize(rank),'chance':chance(len(pool)),
                'candidate_keys':pool,'candidate_pool_sha256':plan['candidate_pool_sha256'],
                'observation_identities':[list(identity(r)) for r in splits['val']],
                'per_subject':{s:summarize(rank[[r.subject_key==s for r in splits['val']]]) for s in sorted({r.subject_key for r in splits['val']})},
                'ranks':rank.tolist(),'selection':{k:v for k,v in selection.items() if k!='grid'},'seconds':time.monotonic()-started}
        write(folder/'validation.json',report);reports.append(report)
        jobs.append({'condition':condition,'status':'validation_complete','selected':report['selection'],'seconds':report['seconds']})
        write(output/'jobs.json',jobs);print(json.dumps({'condition':condition,'metrics':report['metrics'],'selection':report['selection']}),flush=True)
    assert all(r['observation_identities']==reports[0]['observation_identities'] and r['candidate_keys']==pool for r in reports)
    write(output/'validation_summary.json',reports)
    real=reports[0]['metrics']
    gaps=[{'comparison':'real_minus_'+r['condition'],**{k:real[k]-r['metrics'][k] for k in ('recall@1','recall@5','recall@10','mrr')}} for r in reports[1:]]
    write(output/'real_minus_controls.json',gaps)
    decision=useful_validation(reports,cfg['useful_validation_screen']['minimum_mrr_gap_vs_strongest_control'])
    write(output/'decision.json',decision)
    after={p:sha(ROOT/p) for p in before}
    if before!=after:raise RuntimeError('A protected prior artifact changed')
    write(output/'preservation_after.json',{'unchanged':True,'protected_files':len(before),'files_sha256':after})
    print(json.dumps(decision,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--view',type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=4):main(args.output,args.view)
