"""One frozen word-pretraining experiment; training-reader CV only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from eeg_decoding.data.dataset import read_manifest, observation_seed
from eeg_decoding.training.alignment_train import TextCache
from eeg_decoding.evaluation.gru_pilot import SequenceGRU, encode, fit_scaler, reader_folds, collate
from eeg_decoding.evaluation.word_pretrain_v1 import WordEncoder, fit_vocabulary, target_indices, transfer_projection, index_batches, word_ranks
from eeg_decoding.evaluation.gru_retrieval_loss import candidate_cross_entropy
from eeg_decoding.evaluation.spectral_pilot import fit_word_moments, matched_gaussian, count_basis, fit_transform
from eeg_decoding.evaluation.incremental_ridge import ridge_path_predictions
from eeg_decoding.evaluation.controls import shuffled_keys
from eeg_decoding.evaluation.retrieval import ranks, summarize
from run_gru_retrieval_loss import ROOT, SOURCE, sha, write, ident, predict
from prepare_word_pretrain_v1 import CONFIG, preservation_finish

METRICS=('recall@1','recall@5','recall@10','mrr')


def optimizer(model,cfg):
    return torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])


def sentence_stage(model,xs,labels,candidates,steps,cfg,device):
    opt=optimizer(model,cfg); history=[]
    model.train()
    for step,ix in enumerate(index_batches(len(xs),steps,cfg['sentence_batch_size'],cfg['seed']),1):
        x,lengths=collate(xs,ix,device)
        opt.zero_grad(set_to_none=True)
        loss=candidate_cross_entropy(model(x,lengths),candidates,labels[ix],cfg['temperature'])
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip']);opt.step()
        history.append({'step':step,'loss':float(loss.detach()),'examples':len(ix)})
    return history,opt


def word_stage(word,xa,xb,training_labels,query_labels,valid_query,vocabulary,cfg,device):
    x=torch.tensor(np.concatenate(xa),device=device)
    y=torch.tensor(training_labels,dtype=torch.long,device=device)
    assert (y>=0).all() and x.shape[1]==841 and len(x)==len(y)
    opt=optimizer(word,cfg);history=[];word.train()
    for step,ix in enumerate(index_batches(len(x),cfg['word_steps'],cfg['word_batch_size'],cfg['seed']),1):
        opt.zero_grad(set_to_none=True)
        loss=torch.nn.functional.cross_entropy(word(x[ix]),y[ix])
        loss.backward();torch.nn.utils.clip_grad_norm_(word.parameters(),cfg['gradient_clip']);opt.step()
        history.append({'step':step,'loss':float(loss.detach()),'word_slots':len(ix)})
    word.eval();q=np.concatenate(xb);rank=[]
    with torch.no_grad():
        for start in range(0,len(q),cfg['word_batch_size']):
            logits=word(torch.tensor(q[start:start+cfg['word_batch_size']],device=device)).cpu().numpy()
            rank.extend(word_ranks(logits,query_labels[start:start+len(logits)]))
    rank=np.array(rank)
    iv=query_labels>=0
    report={'task':'Word class retrieval, not sentence retrieval','vocabulary_size':len(vocabulary),
        'all_word_slots':summarize(rank),'observed_eeg_slots':summarize(rank[valid_query]),
        'observed_in_vocabulary_slots':summarize(rank[valid_query & iv]) if (valid_query & iv).any() else None,
        'oov_slots':int((~iv).sum()),'missing_eeg_slots':int((~valid_query).sum()),
        'oov_policy':'Every OOV is a miss with reciprocal rank zero; no slot or sentence excluded',
        'ranks':[float(r) if np.isfinite(r) else None for r in rank],
        'query_target_indices':query_labels.tolist(),'query_observed_masks':valid_query.tolist()}
    return history,opt,report


def fit_fold(folder,train,query,wa,wb,la,lb,sa,sb,pool,text,cfg):
    folder.mkdir();device='cuda' if torch.cuda.is_available() else 'cpu'
    vocabulary=fit_vocabulary(la)
    word_train=[w for row in la for w in row];word_query=[w for row in lb for w in row]
    yw=target_indices(word_train,vocabulary);qw=target_indices(word_query,vocabulary)
    assert not (yw<0).any()
    wrong_words=shuffled_keys(vocabulary,cfg['seed']);word_mapping=dict(zip(vocabulary,wrong_words))
    ywrong=target_indices([word_mapping[w] for w in word_train],vocabulary)
    keys=[r.key for r in train];wrong_keys=shuffled_keys(keys,cfg['seed'])
    assert sorted(set(keys))==pool
    lookup={k:i for i,k in enumerate(pool)}
    y=torch.tensor([lookup[k] for k in keys],device=device)
    sy=torch.tensor([lookup[k] for k in wrong_keys],device=device)
    candidates=torch.tensor(text.get(pool),dtype=torch.float32,device=device)
    scaler=fit_scaler(wa);moments=fit_word_moments(wa)
    ga=[matched_gaussian(w,moments,observation_seed(r,cfg['seed'])) for r,w in zip(train,wa)]
    gb=[matched_gaussian(w,moments,observation_seed(r,cfg['seed'])) for r,w in zip(query,wb)]
    assert all(np.array_equal(np.isfinite(a),np.isfinite(b)) for a,b in zip(wa+wb,ga+gb))
    gscaler=fit_scaler(ga)
    np.savez_compressed(folder/'preprocessing.npz',**{'real_'+k:v for k,v in scaler.items()},
        **{'gaussian_'+k:v for k,v in gscaler.items()},**{'moment_'+k:v for k,v in moments.items()})
    write(folder/'vocabulary.json',vocabulary)
    write(folder/'provenance.json',{'fit_observations':[ident(r) for r in train],'query_observations':[ident(r) for r in query],
        'fit_readers':sorted({r.subject for r in train}),'query_readers':sorted({r.subject for r in query}),
        'candidate_keys':pool,'class_shuffle_mapping':dict(zip(keys,wrong_keys)),'word_class_shuffle_mapping':word_mapping,
        'all_learned_fit_scope':'Fold-training readers only, including word vocabulary, scaling and Gaussian moments',
        'gaussian_exact_masks_and_word_counts_match':True,'query_oov_slots':int((qw<0).sum()),
        'query_word_slots':len(qw),'fit_word_slots':len(yw)})
    torch.manual_seed(cfg['seed'])
    base=SequenceGRU(candidates.shape[1],cfg['projection'],cfg['hidden'])
    initial=copy.deepcopy(base.state_dict());torch.save(initial,folder/'sentence_initial.pt')
    torch.manual_seed(cfg['seed'])
    initial_word=WordEncoder(len(vocabulary),cfg['projection'])
    initial_word.project.load_state_dict(base.project.state_dict())
    wi=copy.deepcopy(initial_word.state_dict());torch.save(wi,folder/'word_initial.pt')
    write(folder/'architecture.json',{'sentence_parameters':sum(p.numel() for p in base.parameters()),
        'word_parameters':sum(p.numel() for p in initial_word.parameters()),'word_head_classes':len(vocabulary),
        'transferred_parameters':sum(p.numel() for p in base.project.parameters()),
        'transfer_keys':['project.weight','project.bias'],'input_width':841,'output_width':int(candidates.shape[1])})
    scores=[]
    for arm in cfg['arms']:
        dest=folder/arm;dest.mkdir()
        aa,bb,ss=(ga,gb,gscaler) if arm=='pretrained_gaussian' else (wa,wb,scaler)
        condition='missingness_only' if arm=='pretrained_missingness' else 'real_eeg'
        xa=[encode(w,ss,condition) for w in aa];xb=[encode(w,ss,condition) for w in bb]
        assert [len(x) for x in xa]==[max(1,len(w)) for w in wa]
        sentence=SequenceGRU(candidates.shape[1],cfg['projection'],cfg['hidden']).to(device)
        sentence.load_state_dict(initial)
        stage_one_steps=0;word_report=None
        if arm.startswith('pretrained_'):
            word=WordEncoder(len(vocabulary),cfg['projection']).to(device);word.load_state_dict(wi)
            valid_query=np.concatenate([np.isfinite(w).all(1) for w in wb])
            history,opt,word_report=word_stage(word,xa,xb,ywrong if arm=='pretrained_class_shuffled' else yw,
                qw,valid_query,vocabulary,cfg,device)
            write(dest/'word_history.json',history);write(dest/'word_holdout.json',word_report)
            torch.save({'model':word.state_dict(),'optimizer':opt.state_dict(),'steps':cfg['word_steps']},dest/'word_final.pt')
            transfer_projection(word,sentence)
            transfer_checks={k:bool(torch.equal(v,word.project.state_dict()[k.split('.',1)[1]])) if k.startswith('project.')
                else bool(torch.equal(v.cpu(),initial[k])) for k,v in sentence.state_dict().items()}
            assert all(transfer_checks.values())
            write(dest/'transfer_verification.json',{'passed':True,'only_projection_transferred':True,'tensor_checks':transfer_checks})
            stage_one_steps=cfg['word_steps']
            del word,opt
        elif arm=='fresh_equal_steps':
            history,opt=sentence_stage(sentence,xa,y,candidates,cfg['word_steps'],cfg,device)
            write(dest/'extra_sentence_history.json',history)
            torch.save({'model':sentence.state_dict(),'steps':cfg['word_steps']},dest/'extra_sentence_final.pt')
            stage_one_steps=cfg['word_steps'];del opt
        # No word classifier or word target is used by the sentence encoder.
        torch.save(sentence.state_dict(),dest/'sentence_stage_start.pt')
        history,opt=sentence_stage(sentence,xa,sy if arm=='pretrained_class_shuffled' else y,
            candidates,cfg['sentence_steps'],cfg,device)
        write(dest/'sentence_history.json',history)
        torch.save({'model':sentence.state_dict(),'optimizer':opt.state_dict(),'steps':cfg['sentence_steps']},dest/'sentence_final.pt')
        prediction=predict(sentence,xb,device,cfg['sentence_batch_size'])
        rank=ranks(prediction,text.get(pool),[r.key for r in query],pool)
        score={'arm':arm,'evaluation':'training_reader_holdout','stage_one_updates':stage_one_steps,
            'sentence_updates':cfg['sentence_steps'],'total_updates':stage_one_steps+cfg['sentence_steps'],
            **summarize(rank),'ranks':rank.tolist()}
        write(dest/'sentence_holdout.json',score);scores.append(score)
        print(folder.name,arm,'complete:',score['recall@1'],score['mrr'],flush=True)
        del sentence,opt
    # Same structural observations and fixed training-only alpha as earlier pilots.
    counts_a=np.array([len(w) for w in wa]);counts_b=np.array([len(w) for w in wb])
    ba,_=count_basis(counts_a,counts_a);bb,_=count_basis(counts_a,counts_b)
    a,b,transform=fit_transform(np.c_[sa,ba],np.c_[sb,bb],'median','standard')
    _,pred,fitted=next(ridge_path_predictions(a,b,text.get(keys).astype(float),[cfg['structural_alpha']]))
    np.savez_compressed(folder/'structural_ridge.npz',**fitted,fill=transform['fill'],center=transform['center'],scale=transform['scale'])
    rank=ranks(pred,text.get(pool),[r.key for r in query],pool)
    scores.append({'arm':'structural_only','evaluation':'training_reader_holdout',**summarize(rank),'ranks':rank.tolist()})
    write(folder/'sentence_scores.json',scores)
    return scores


def main(output):
    plan=json.loads((output/'plan.json').read_text());cfg=plan['configuration']
    if (output/'fit_started.json').exists(): raise FileExistsError('No rerunning or overwriting a started fit')
    assert json.loads((output/'feasibility.json').read_text())['feasible']
    assert sha(CONFIG)==plan['configuration_sha256']
    for p,h in plan['implementation_sha256'].items():
        assert sha(ROOT/p)==h,'Frozen implementation changed: '+p
    assert sha(SOURCE/'word_sequences.npz')==plan['source_cache_sha256']
    assert sha(plan['manifest'])==plan['manifest_sha256'] and sha(plan['text_cache'])==plan['text_cache_sha256']
    feasibility=json.loads((output/'feasibility.json').read_text())
    assert sha(output/'exported_word_targets.json')==feasibility['word_label_cache_sha256']
    write(output/'fit_started.json',{'configuration_sha256':sha(CONFIG),'fixed_final_stage_steps':cfg['sentence_steps'],'no_selection':True})
    records={tuple(ident(r)):r for r in read_manifest(plan['manifest'])}
    train=[records[tuple(k)] for k in plan['train_observations']]
    labels=json.loads((output/'exported_word_targets.json').read_text())
    with np.load(SOURCE/'word_sequences.npz',allow_pickle=False) as z:
        words,offsets=z['train_words'],z['train_offsets']
    seq=[words[a:b] for a,b in zip(offsets[:-1],offsets[1:])]
    assert len(labels)==len(seq)==len(train)==2245 and all(len(w)==len(l) for w,l in zip(seq,labels))
    with np.load(ROOT/'data/processed/zuco1_trt_mean_pilot_v1_20261030/sentence_vectors.npz',allow_pickle=False) as z:
        structure=z['count_duration_only_train'][:,:14]
    text=TextCache(plan['text_cache']);assert text.digest==plan['text_cache_sha256']
    table=[];folds=reader_folds([r.subject for r in train])
    for reader,a,b in folds:
        start=time.monotonic()
        scores=fit_fold(output/('cv_'+reader),[train[i] for i in a],[train[i] for i in b],
            [seq[i] for i in a],[seq[i] for i in b],[labels[i] for i in a],[labels[i] for i in b],
            structure[a],structure[b],plan['training_candidate_keys'],text,cfg)
        table += [dict(score,reader=reader) for score in scores]
        write(output/'training_cv_all_scores.json',table)
        print('FOLD COMPLETE',reader,round(time.monotonic()-start,1),'seconds',flush=True)
    summary={}
    for arm in cfg['arms']+['structural_only']:
        rows=[r for r in table if r['arm']==arm];assert len(rows)==8
        summary[arm]={'macro':{m:float(np.mean([r[m] for r in rows])) for m in METRICS},
            'per_reader':{r['reader']:{m:r[m] for m in (*METRICS,'n_queries')} for r in rows}}
    write(output/'training_cv_summary.json',summary)
    real=summary['pretrained_real']
    gaps={arm:{'macro_difference':{m:real['macro'][m]-s['macro'][m] for m in METRICS},
        'per_reader_difference':{reader:{m:real['per_reader'][reader][m]-s['per_reader'][reader][m] for m in METRICS} for reader,_,_ in folds}}
        for arm,s in summary.items() if arm!='pretrained_real'}
    checks={arm:(g['macro_difference']['recall@1']>=cfg['gate']['minimum_recall1_gap'] and
        g['macro_difference']['mrr']>=cfg['gate']['minimum_mrr_gap'] and
        sum(d['recall@1']>0 for d in g['per_reader_difference'].values())>=cfg['gate']['minimum_positive_readers']) for arm,g in gaps.items()}
    write(output/'real_minus_controls.json',gaps)
    write(output/'decision.json',{'descriptive_gate_passed':all(checks.values()),'checks':checks,
        'next_action':'Stop after fixed training CV; no expansion or outer/test evaluation','outer_evaluated':False,'test_evaluated':False})
    write(output/'access_summary.json',{'word_cache_members':['train_words','train_offsets'],'structural_members':['count_duration_only_train'],
        'source_readers_during_feasibility':plan['readers'],'raw_sources_during_fit':False,'outer_eeg_arrays_loaded':False,
        'test_eeg_arrays_loaded':False,'outer_evaluated':False,'test_evaluated':False})
    preservation_finish(output)
    write(output/'fit_complete.json',{'complete':True,'folds':8,'sentence_holdout_rows':len(table)})
    print('COMPLETE; training-only gate',all(checks.values()),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=4):main(args.output.resolve())
