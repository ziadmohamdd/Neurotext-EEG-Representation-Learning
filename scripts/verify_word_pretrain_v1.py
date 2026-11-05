"""Replay final checkpoints using training-cache members only; verify provenance."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from eeg_decoding.data.dataset import read_manifest, observation_seed
from eeg_decoding.evaluation.gru_pilot import SequenceGRU, encode, reader_folds, fit_scaler
from eeg_decoding.evaluation.word_pretrain_v1 import fit_vocabulary, target_indices, WordEncoder, word_ranks
from eeg_decoding.evaluation.spectral_pilot import matched_gaussian, fit_word_moments
from eeg_decoding.evaluation.retrieval import ranks, summarize
from eeg_decoding.training.alignment_train import TextCache
from run_gru_retrieval_loss import ROOT, SOURCE, sha, write, ident, predict


def read(path): return json.loads(path.read_text())


def main(output):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    plan=read(output/'plan.json');cfg=plan['configuration'];assert read(output/'fit_complete.json')['complete']
    for p,h in plan['implementation_sha256'].items(): assert sha(ROOT/p)==h
    assert sha(SOURCE/'word_sequences.npz')==plan['source_cache_sha256']
    labels=read(output/'exported_word_targets.json');records={tuple(ident(r)):r for r in read_manifest(plan['manifest'])}
    rows=[records[tuple(k)] for k in plan['train_observations']];text=TextCache(plan['text_cache'])
    with np.load(SOURCE/'word_sequences.npz',allow_pickle=False) as z: words,offsets=z['train_words'],z['train_offsets']
    seq=[words[a:b] for a,b in zip(offsets[:-1],offsets[1:])]; pool=plan['training_candidate_keys']
    all_queries=set(); sentence_count=0;word_count=0;transfer_count=0
    summary=read(output/'training_cv_summary.json')
    for reader,a,b in reader_folds([r.subject for r in rows]):
        folder=output/('cv_'+reader);prov=read(folder/'provenance.json')
        aset={tuple(k) for k in prov['fit_observations']};bset={tuple(k) for k in prov['query_observations']}
        assert not aset & bset and aset|bset=={tuple(k) for k in plan['train_observations']}
        assert not all_queries & bset;all_queries |= bset
        assert not set(prov['fit_readers']) & set(prov['query_readers']) and prov['candidate_keys']==pool
        vocabulary=read(folder/'vocabulary.json')
        assert vocabulary==fit_vocabulary([labels[i] for i in a])
        qw=target_indices([w for i in b for w in labels[i]],vocabulary)
        assert prov['query_oov_slots']==int((qw<0).sum())
        with np.load(folder/'preprocessing.npz',allow_pickle=False) as z: pp={k:z[k] for k in z.files}
        # Rebuild every word-level fitted quantity from seven fit readers only.
        expected_scaler=fit_scaler([seq[i] for i in a])
        expected_moments=fit_word_moments([seq[i] for i in a])
        for k,v in expected_scaler.items(): np.testing.assert_array_equal(v,pp['real_'+k])
        for k,v in expected_moments.items(): np.testing.assert_array_equal(v,pp['moment_'+k])
        gaussian_fit=[matched_gaussian(seq[i],expected_moments,observation_seed(rows[i],cfg['seed'])) for i in a]
        for k,v in fit_scaler(gaussian_fit).items(): np.testing.assert_array_equal(v,pp['gaussian_'+k])
        del gaussian_fit
        assert set(prov['word_class_shuffle_mapping'])==set(vocabulary)
        assert all(k!=v for k,v in prov['word_class_shuffle_mapping'].items())
        assert all(k!=v for k,v in prov['class_shuffle_mapping'].items())
        initial=torch.load(folder/'sentence_initial.pt',map_location='cpu',weights_only=True)
        old_initial=torch.load(ROOT/f'results/gru_pilot_v1_20261101/cv_{reader}/initial_weights.pt',map_location='cpu',weights_only=True)
        for k in initial: torch.testing.assert_close(initial[k],old_initial[k],rtol=0,atol=0)
        for arm in cfg['arms']:
            dest=folder/arm
            if arm=='pretrained_gaussian':
                moments={k:pp['moment_'+k] for k in ('count','mean','sd')}
                query=[matched_gaussian(seq[i],moments,observation_seed(rows[i],cfg['seed'])) for i in b]
                assert all(np.array_equal(np.isfinite(w),np.isfinite(seq[i])) for i,w in zip(b,query))
                scaler={k:pp['gaussian_'+k] for k in ('center','scale')}
            else:
                query=[seq[i] for i in b];scaler={k:pp['real_'+k] for k in ('center','scale')}
            xq=[encode(w,scaler,'missingness_only' if arm=='pretrained_missingness' else 'real_eeg') for w in query]
            start=torch.load(dest/'sentence_stage_start.pt',map_location='cpu',weights_only=True)
            if arm.startswith('pretrained_'):
                word_state=torch.load(dest/'word_final.pt',map_location='cpu',weights_only=True)['model']
                for k in initial:
                    target=word_state[k] if k.startswith('project.') else initial[k]
                    torch.testing.assert_close(start[k],target,rtol=0,atol=0)
                transfer_count+=1
                assert len(read(dest/'word_history.json'))==cfg['word_steps']
                word=WordEncoder(len(vocabulary),cfg['projection']).to(device);word.load_state_dict(word_state);word.eval()
                flattened=np.concatenate(xq);wr=[]
                with torch.no_grad():
                    for j in range(0,len(flattened),256):
                        logits=word(torch.tensor(flattened[j:j+256],device=device)).cpu().numpy()
                        wr.extend(word_ranks(logits,qw[j:j+len(logits)]))
                saved=read(dest/'word_holdout.json')
                expected=np.array([np.inf if r is None else r for r in saved['ranks']])
                np.testing.assert_array_equal(wr,expected)
                for m in ('recall@1','recall@5','recall@10','mrr'):
                    assert abs(summarize(wr)[m]-saved['all_word_slots'][m])<1e-12
                word_count+=len(wr)
            elif arm=='fresh_same_sentence':
                for k in initial: torch.testing.assert_close(start[k],initial[k],rtol=0,atol=0)
            else:
                extra=torch.load(dest/'extra_sentence_final.pt',map_location='cpu',weights_only=True)['model']
                for k in initial: torch.testing.assert_close(start[k],extra[k],rtol=0,atol=0)
                assert len(read(dest/'extra_sentence_history.json'))==cfg['word_steps']
            history=read(dest/'sentence_history.json')
            assert len(history)==cfg['sentence_steps'] and all(np.isfinite(h['loss']) for h in history)
            model=SequenceGRU(384,cfg['projection'],cfg['hidden']).to(device)
            model.load_state_dict(torch.load(dest/'sentence_final.pt',map_location='cpu',weights_only=True)['model'])
            prediction=predict(model,xq,device,cfg['sentence_batch_size'])
            rank=ranks(prediction,text.get(pool),[rows[i].key for i in b],pool)
            saved=read(dest/'sentence_holdout.json');np.testing.assert_array_equal(rank,saved['ranks'])
            assert len(rank)==len(b) and saved['total_updates']==(256 if arm=='fresh_same_sentence' else 384)
            for m in ('recall@1','recall@5','recall@10','mrr'):
                assert abs(summarize(rank)[m]-summary[arm]['per_reader'][reader][m])<1e-12
            sentence_count+=len(rank)
        previous=read(ROOT/f'results/gru_pilot_v1_20261101/cv_{reader}/scores.json')
        original=next(r for r in previous if r['condition']=='structural_only')
        current=next(r for r in read(folder/'sentence_scores.json') if r['arm']=='structural_only')
        np.testing.assert_array_equal(original['ranks'],current['ranks'])
        print('VERIFIED',reader,flush=True)
    assert len(all_queries)==2245
    for arm,row in summary.items():
        for m in row['macro']:
            assert abs(row['macro'][m]-np.mean([r[m] for r in row['per_reader'].values()]))<1e-12
    assert read(output/'preservation_after.json')['unchanged']
    access=read(output/'access_summary.json')
    assert not access['outer_eeg_arrays_loaded'] and not access['test_eeg_arrays_loaded']
    result={'passed':True,'folds':8,'unique_training_holdouts':len(all_queries),'replayed_sentence_ranks':sentence_count,
        'replayed_word_ranks':word_count,'projection_only_transfers_verified':transfer_count,
        'initial_sentence_weights_match_prior_gru':True,'structural_ranks_match_prior_gru':True,
        'vocabularies_rebuilt_from_fold_training_labels_only':True,'preprocessing_rebuilt_from_fold_training_words_only':True,
        'reader_separation_and_fixed_budgets_verified':True}
    write(output/'evidence_verification.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);main(parser.parse_args().output.resolve())
