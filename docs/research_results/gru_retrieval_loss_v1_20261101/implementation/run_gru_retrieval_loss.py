"""Loss-only GRU comparison; outer validation and test arrays are never loaded."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import copy
import hashlib
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from eeg_decoding.data.dataset import read_manifest, observation_seed
from eeg_decoding.training.alignment_train import TextCache
from eeg_decoding.evaluation.gru_pilot import SequenceGRU, encode, fit_scaler, collate, reader_folds
from eeg_decoding.evaluation.controls import shuffled_keys
from eeg_decoding.evaluation.gru_retrieval_loss import candidate_cross_entropy
from eeg_decoding.evaluation.spectral_pilot import fit_word_moments, matched_gaussian, count_basis, fit_transform
from eeg_decoding.evaluation.incremental_ridge import ridge_path_predictions
from eeg_decoding.evaluation.retrieval import ranks, summarize

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/gru_retrieval_loss_v1_20261101.json'
SOURCE = ROOT / 'results/word_order_pilot_v1_20261031'
METRICS = ('recall@1', 'recall@5', 'recall@10', 'mrr')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def ident(r):
    return [r.dataset, r.subject, r.session, r.task, r.sentence_id]


def predict(model, xs, device, batch):
    model.eval(); result = []
    with torch.no_grad():
        for start in range(0, len(xs), batch):
            x, n = collate(xs, list(range(start, min(start+batch, len(xs)))), device)
            result.append(model(x, n).cpu().numpy())
    return np.concatenate(result)


def fit_fold(folder, train, query, sequences_a, sequences_b, structure_a, structure_b, pool, text, cfg, epochs, checkpoints):
    folder.mkdir()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    moments = fit_word_moments(sequences_a)
    noise_a = [matched_gaussian(w, moments, observation_seed(r, cfg['seed'])) for r,w in zip(train,sequences_a)]
    noise_b = [matched_gaussian(w, moments, observation_seed(r, cfg['seed'])) for r,w in zip(query,sequences_b)]
    assert all(np.array_equal(np.isfinite(w), np.isfinite(n)) for w,n in zip(sequences_a+sequences_b,noise_a+noise_b))
    scaler = fit_scaler(sequences_a); noise_scaler = fit_scaler(noise_a)
    np.savez_compressed(folder/'preprocessing.npz', **{'real_'+k:v for k,v in scaler.items()},
                        **{'gaussian_'+k:v for k,v in noise_scaler.items()}, **{'moments_'+k:v for k,v in moments.items()})
    keys = [r.key for r in train]; wrong = shuffled_keys(keys, cfg['seed'])
    training_pool = sorted(set(keys))
    assert training_pool == pool, 'All candidates must be unique fold-training classes'
    lookup = {key:i for i,key in enumerate(training_pool)}
    candidate_vectors = torch.as_tensor(text.get(training_pool), dtype=torch.float32, device=device)
    write(folder/'provenance.json', {'fit_observations': [ident(r) for r in train], 'query_observations':[ident(r) for r in query],
        'fit_readers':sorted({r.subject for r in train}), 'query_readers':sorted({r.subject for r in query}),
        'candidate_keys':pool, 'class_shuffle_mapping':dict(zip(keys,wrong)), 'gaussian_exact_masks_match':True,
        'fit_scope':'All moments, scaling and imputation use fit_observations only',
        'query_classes_absent_from_fit':sorted({r.key for r in query}-set(keys))})
    torch.manual_seed(cfg['seed'])
    base = SequenceGRU(text.get(pool).shape[1], cfg['projection'], cfg['hidden'])
    initial = copy.deepcopy(base.state_dict()); torch.save(initial, folder/'initial_weights.pt')
    write(folder/'architecture.json', {'parameters':sum(p.numel() for p in base.parameters()),
        'layers':{name:sum(p.numel() for p in module.parameters()) for name,module in [('projection',base.project),('gru',base.gru),('head',base.head)]},
        'output_width':text.get(pool).shape[1], 'input_width':841})
    results = []
    for condition in cfg['conditions']:
        dest = folder/condition; dest.mkdir()
        aa,bb,ss = (noise_a,noise_b,noise_scaler) if condition == 'matched_gaussian' else (sequences_a,sequences_b,scaler)
        xa = [encode(w,ss,condition,observation_seed(r,cfg['seed']+1)) for r,w in zip(train,aa)]
        xb = [encode(w,ss,condition,observation_seed(r,cfg['seed']+1)) for r,w in zip(query,bb)]
        target_keys = wrong if condition=='class_shuffled' else keys
        labels = torch.as_tensor([lookup[k] for k in target_keys], dtype=torch.long, device=device)
        targets = torch.as_tensor(text.get(target_keys), dtype=torch.float32, device=device)
        model = SequenceGRU(targets.shape[1],cfg['projection'],cfg['hidden']).to(device)
        model.load_state_dict(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        rng = np.random.default_rng(cfg['seed']); history=[]
        for epoch in range(1,epochs+1):
            model.train(); total=0.; order=rng.permutation(len(train))
            for start in range(0,len(order),cfg['batch_size']):
                ix=order[start:start+cfg['batch_size']]
                x,n=collate(xa,ix,device); optimizer.zero_grad(set_to_none=True)
                loss=candidate_cross_entropy(model(x,n),candidate_vectors,labels[ix],cfg['temperature'])
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip']); optimizer.step()
                total+=float(loss.detach())*len(ix)
            history.append({'epoch':epoch,'training_candidate_ce':total/len(train)})
            if epoch in checkpoints:
                pred=predict(model,xb,device,cfg['batch_size'])
                rank=ranks(pred,text.get(pool),[r.key for r in query],pool)
                results.append({'condition':condition,'epoch':epoch,**summarize(rank),'ranks':rank.tolist(),
                    'per_reader':{s:summarize(rank[[r.subject==s for r in query]]) for s in sorted({r.subject for r in query})}})
                torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch},dest/f'epoch_{epoch}.pt')
        write(dest/'history.json',history)
        print(folder.name,condition,'completed',epochs,'epochs',flush=True)
    # Existing structural recipe; alpha=1 fixed from preceding training-only semantic CV.
    counts_a=np.array([len(w) for w in sequences_a]); counts_b=np.array([len(w) for w in sequences_b])
    ba,_=count_basis(counts_a,counts_a); bb,_=count_basis(counts_a,counts_b)
    a,b,transform=fit_transform(np.c_[structure_a,ba],np.c_[structure_b,bb],'median','standard')
    _,pred,fit=next(ridge_path_predictions(a,b,text.get(keys).astype(float),[cfg['structural_alpha']]))
    np.savez_compressed(folder/'structural_ridge.npz',**fit,fill=transform['fill'],center=transform['center'],scale=transform['scale'])
    rank=ranks(pred,text.get(pool),[r.key for r in query],pool)
    results.append({'condition':'structural_only','epoch':0,**summarize(rank),'ranks':rank.tolist(),
        'per_reader':{s:summarize(rank[[r.subject==s for r in query]]) for s in sorted({r.subject for r in query})}})
    write(folder/'scores.json',results)
    return results


def main(output):
    if output.exists(): raise FileExistsError('Fresh versioned output required')
    cfg=json.loads(CONFIG.read_text()); old=json.loads((SOURCE/'plan.json').read_text())
    view=json.loads((SOURCE/'feature_view.json').read_text())
    assert sha(SOURCE/'word_sequences.npz')==view['cache_sha256']
    # Hash prior artifacts, code, targets and prepared caches; never open raw/test sources.
    protected=[p for d in ('results','docs','checkpoints','configs','src','scripts','tests') for p in (ROOT/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    protected += [p for p in (ROOT/'data/processed/zuco1_trt_mean_pilot_v1_20261030').glob('*') if p.is_file()]
    protected += [ROOT/'README.md',Path(old['manifest']),Path(old['text_cache'])]
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    raw_stats={str(p.relative_to(ROOT)):[p.stat().st_size,p.stat().st_mtime_ns] for p in (ROOT/'data').rglob('*') if p.is_file()}
    output.mkdir(parents=True)
    write(output/'preservation_before.json',before); write(output/'raw_metadata_before.json',raw_stats)
    (output/'git_status_before.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True))
    write(output/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
        'cuda':torch.cuda.is_available(),'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})
    write(output/'plan.json',{'configuration':cfg,'configuration_sha256':sha(CONFIG),'source_cache_sha256':view['cache_sha256'],
        'source_provenance':view['sources'],'train_observations':old['train_observations'],'validation_observations':old['validation_observations'],
        'training_candidate_keys':old['inner_candidate_keys'],'outer_candidate_keys':old['candidate_keys'],
        'outer_candidate_pool_sha256':old['candidate_pool_sha256'],'manifest_sha256':sha(old['manifest']),
        'text_cache_sha256':sha(old['text_cache']),'authorization':'User requested an alternative; prediction-concentration audit motivated a loss-only comparison; no outer validation or test evaluation',
        'limitations':old['limitations']})
    for p in [CONFIG,Path(__file__),ROOT/'src/eeg_decoding/evaluation/gru_pilot.py',ROOT/'tests/test_gru_pilot.py',
              ROOT/'src/eeg_decoding/evaluation/spectral_pilot.py',ROOT/'src/eeg_decoding/evaluation/retrieval.py',ROOT/'src/eeg_decoding/evaluation/controls.py',ROOT/'src/eeg_decoding/evaluation/incremental_ridge.py',ROOT/'src/eeg_decoding/evaluation/gru_retrieval_loss.py',ROOT/'tests/test_gru_retrieval_loss.py',ROOT/'scripts/prepare_gru_loss_comparison.py']:
        (output/'implementation').mkdir(exist_ok=True); shutil.copyfile(p,output/'implementation'/p.name)
    records={tuple(ident(r)):r for r in read_manifest(old['manifest'])}
    train=[records[tuple(k)] for k in old['train_observations']]
    text=TextCache(old['text_cache']); assert text.digest==old['text_cache_sha256']
    with np.load(SOURCE/'word_sequences.npz',allow_pickle=False) as z:
        words,offsets=z['train_words'],z['train_offsets']
    seq=[words[a:b] for a,b in zip(offsets[:-1],offsets[1:])]
    with np.load(ROOT/'data/processed/zuco1_trt_mean_pilot_v1_20261030/sentence_vectors.npz',allow_pickle=False) as z:
        structural=z['count_duration_only_train'][:,:14]
    assert len(train)==len(seq)==len(structural)==2245
    folds=reader_folds([r.subject for r in train]); table=[]
    write(output/'coverage.json',{'observations':len(seq),'word_slots':len(words),'complete_words':int(np.isfinite(words).all(1).sum()),
        'all_missing_observations':[ident(r) for r,w in zip(train,seq) if not np.isfinite(w).any()]})
    for reader,a,b in folds:
        start=time.monotonic()
        rows=fit_fold(output/('cv_'+reader),[train[i] for i in a],[train[i] for i in b],[seq[i] for i in a],[seq[i] for i in b],
            structural[a],structural[b],old['inner_candidate_keys'],text,cfg,cfg['epochs'],cfg['epoch_candidates'])
        table += [dict(row,reader=reader) for row in rows]
        write(output/'training_cv_all_scores.json',table)
        print('FOLD COMPLETE',reader,round(time.monotonic()-start,1),'seconds',flush=True)
    epoch=max(cfg['epoch_candidates'],key=lambda e:(np.mean([r['recall@1'] for r in table if r['condition']=='real_eeg' and r['epoch']==e]),-e))
    write(output/'frozen_selection.json',{'common_epoch':epoch,'rule':cfg['epoch_selection'],'outer_arrays_loaded':False})
    by={}
    for c in cfg['conditions']+['structural_only']:
        rows=[r for r in table if r['condition']==c and r['epoch']==(0 if c=='structural_only' else epoch)]
        assert len(rows)==8
        by[c]={'macro':{m:float(np.mean([r[m] for r in rows])) for m in METRICS},'per_reader':{r['reader']:{m:r[m] for m in METRICS} for r in rows}}
    write(output/'training_cv_summary.json',by)
    gaps={c:{'macro_difference':{m:by['real_eeg']['macro'][m]-row['macro'][m] for m in METRICS},
        'per_reader_difference':{s:{m:by['real_eeg']['per_reader'][s][m]-row['per_reader'][s][m] for m in METRICS} for s,_,_ in folds}}
        for c,row in by.items() if c!='real_eeg'}
    checks={c:(g['macro_difference']['recall@1']>=cfg['gate']['minimum_recall1_gap'] and
        g['macro_difference']['mrr']>=cfg['gate']['minimum_mrr_gap'] and
        sum(d['recall@1']>0 for d in g['per_reader_difference'].values())>=cfg['gate']['minimum_positive_readers']) for c,g in gaps.items()}
    passed=all(checks.values())
    write(output/'real_minus_controls.json',gaps)
    write(output/'decision.json',{'training_gate_passed':passed,'checks':checks,'outer_validation_permitted':False,'test_evaluated':False})
    # This bounded loss-only experiment stops on training CV, regardless of gate.
    changed=[p for p,h in before.items() if sha(ROOT/p)!=h]
    raw_changed=[p for p,v in raw_stats.items() if [(ROOT/p).stat().st_size,(ROOT/p).stat().st_mtime_ns]!=v]
    write(output/'preservation_after.json',{'unchanged':not changed and not raw_changed,'hashed_files':len(before),'raw_metadata_files':len(raw_stats),'changed':changed,'raw_metadata_changed':raw_changed})
    write(output/'access_summary.json',{'word_cache_members_read':['train_words','train_offsets'],
        'structural_members_read':['count_duration_only_train'],
        'raw_source_files_opened':False,'test_arrays_loaded':False,'test_evaluated':False,'outer_evaluated':False})
    assert not changed and not raw_changed
    print(json.dumps({'common_epoch':epoch,'training_cv':by,'gate_passed':passed}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=4): main(args.output.resolve())
