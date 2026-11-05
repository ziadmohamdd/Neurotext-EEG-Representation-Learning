import numpy as np
import pytest
import torch
from scipy.io import savemat
from eeg_decoding.data.word_targets_v1 import iter_labelled_training_slots, check_pairing
from eeg_decoding.evaluation.word_pretrain_v1 import WordEncoder, transfer_projection, fit_vocabulary, target_indices, index_batches, word_ranks
from eeg_decoding.evaluation.gru_pilot import SequenceGRU, encode, fit_scaler, reader_folds
from eeg_decoding.evaluation.spectral_pilot import fit_word_moments, matched_gaussian


def test_export_pairing_and_skipped_slots(tmp_path):
    names = ['content', *['TRT_'+b for b in ('t1','t2','a1','a2','b1','b2','g1','g2')], 'rawET']
    words = np.empty((1,2), dtype=[(n,object) for n in names])
    for i,label in enumerate(['Exact', 'word,']):
        words['content'][0,i] = label
        for n in names[1:-1]: words[n][0,i] = np.full((1,105), i+1.)
        words['rawET'][0,i] = np.ones((4,31))
    sentences = np.empty((1,2), dtype=[('content',object),('word',object),('rawData',object)])
    sentences[0,0] = ('Exact word,', words, np.ones((105,100)))
    sentences[0,1] = ('Skipped', np.empty((0,0)), np.ones((105,100)))
    path=tmp_path/'training.mat'; savemat(path,{'sentenceData':sentences},do_compression=True)
    slot,parent,labels,eeg = list(iter_labelled_training_slots(path,{0}))[0]
    assert slot==0 and labels==['Exact','word,'] and eeg.shape==(2,840)
    check_pairing(parent,labels,eeg,'Exact word,',eeg.copy())
    with pytest.raises(ValueError): check_pairing(parent,labels,eeg,'Exact word.',eeg)
    with pytest.raises(ValueError): check_pairing(parent,labels,eeg,'Exact word,',eeg[::-1])


def test_training_vocabulary_unchanged_strings_and_oov():
    vocab=fit_vocabulary([['Exact','word,'],['Exact','WORD']])
    assert vocab==['Exact','WORD','word,']
    labels=target_indices(['new','word,'],vocab)
    assert labels.tolist()==[-1,2] and 'new' not in vocab
    rank=word_ranks(np.array([[0.,1.,2.],[0.,1.,2.]]),labels)
    assert np.isinf(rank[0]) and rank[1]==1


def test_transfer_only_projection():
    torch.manual_seed(42); sentence=SequenceGRU(); word=WordEncoder(7)
    original={k:v.clone() for k,v in sentence.state_dict().items()}
    transfer_projection(word,sentence)
    for k,v in sentence.state_dict().items():
        expected=word.state_dict()[k] if k.startswith('project.') else original[k]
        torch.testing.assert_close(v,expected,rtol=0,atol=0)
    x=torch.randn(5,841)
    torch.testing.assert_close(torch.tanh(sentence.project(x)),torch.tanh(word.project(x)))
    with pytest.raises(ValueError): word(torch.randn(5,842))


def test_masks_gaussian_and_training_only_scaler():
    w=np.random.default_rng(42).uniform(0,3,(4,840)); w[1]=np.nan
    scaler=fit_scaler([w]); before={k:v.copy() for k,v in scaler.items()}
    q=w*1000; x=encode(q,scaler)
    for k in before: np.testing.assert_array_equal(scaler[k],before[k])
    noise=matched_gaussian(w,fit_word_moments([w]),42)
    np.testing.assert_array_equal(np.isfinite(noise),np.isfinite(w))
    missing=encode(w,scaler,'missingness_only')
    assert not missing[:,:840].any() and missing.shape==x.shape
    np.testing.assert_array_equal(missing[:,-1],x[:,-1])


def test_fixed_budget_and_grouping():
    batches=list(index_batches(13,9,5,42))
    assert len(batches)==9 and all(1<=len(b)<=5 for b in batches)
    for a,b in zip(batches,index_batches(13,9,5,42)): np.testing.assert_array_equal(a,b)
    subjects=np.repeat(['ZDM','ZDN','ZGW','ZJM','ZJN','ZKH','ZMG','ZPH'],3)
    for reader,a,b in reader_folds(subjects):
        assert not set(a)&set(b) and reader not in subjects[a] and set(subjects[b])=={reader}
