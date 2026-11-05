"""Synthetic gradient/protocol tests, not scientific decoding evidence."""
import json
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from eeg_decoding.models.projection import alignment_loss
from eeg_decoding.training.checkpoints import load_checkpoint
from eeg_decoding.workflow import train_neural, evaluate_checkpoint


def test_training_bank_keeps_reverse_term_and_receives_no_gradient():
    torch.manual_seed(11)
    query=torch.randn(3,5,requires_grad=True)
    bank=torch.randn(4,5,requires_grad=True)
    batch_keys=['b','b','a']; bank_keys=['a','b','c','d']
    target=bank.detach()[[1,1,0]]
    q=F.normalize(query,dim=-1); t=F.normalize(target,dim=-1)
    scores=q @ t.T/.07
    positive=torch.tensor([[x==y for y in batch_keys] for x in batch_keys])
    reverse=-(torch.logsumexp(scores.T.masked_fill(~positive.T,-torch.inf),1)-torch.logsumexp(scores.T,1)).mean()
    expected=(F.cross_entropy(q @ F.normalize(bank.detach(),dim=-1).T/.07,torch.tensor([1,1,0]))+reverse)/2
    loss=alignment_loss(query,target,batch_keys,candidates=bank,candidate_keys=bank_keys)
    torch.testing.assert_close(loss,expected)
    loss.backward()
    assert query.grad.abs().sum()>0 and bank.grad is None
    for bad in [bank_keys[:-1],['a','a','c','d'],['a','x','c','d']]:
        with pytest.raises(ValueError):
            alignment_loss(query,target,batch_keys,candidates=bank,candidate_keys=bad)


def test_bank_equal_to_unique_batch_preserves_existing_loss_and_gradient():
    torch.manual_seed(4)
    q=torch.randn(4,7,requires_grad=True); t=torch.randn(4,7); keys=['a','b','c','d']
    old=alignment_loss(q,t,keys)
    old_grad=torch.autograd.grad(old,q,retain_graph=True)[0]
    new=alignment_loss(q,t,keys,candidates=t,candidate_keys=keys)
    torch.testing.assert_close(old,new)
    torch.testing.assert_close(old_grad,torch.autograd.grad(new,q)[0])


@pytest.mark.parametrize('control',['real','gaussian','shuffled'])
def test_fixed_cnn_comparison_has_training_only_bank_and_common_validation(records,config,text_cache,tmp_path,control):
    from eeg_decoding.utils.reproducibility import seed_all
    from eeg_decoding.workflow import make_splits,loader
    from eeg_decoding.preprocessing.normalization import ChannelNormalizer
    cfg=dict(config,epochs=1)
    checkpoints=[]
    for scope in ['batch','training_unique']:
        out=tmp_path/scope
        train_neural(records,dict(cfg,alignment_candidate_scope=scope),'alignment',out,text_cache.path,control=control)
        m=load_checkpoint(out/'best.pt')
        checkpoints.append(m)
        result=evaluate_checkpoint(records,out/'best.pt',text_cache.path,out/'evaluation.json')
        assert result['metrics']['n_queries']==10
        assert result['evaluation_control']==('gaussian' if control=='gaussian' else 'real')
    baseline,full=checkpoints
    assert baseline['split_digest']==full['split_digest'] and baseline['normalizer']==full['normalizer']
    assert baseline['initial_model_sha256']==full['initial_model_sha256']
    assert full['alignment_objective']['candidate_keys']==full['splits']['splits']['train']['sentence_keys']
    assert len(full['alignment_objective']['candidate_keys'])==10
    assert full['alignment_objective']['model_selection']==baseline['alignment_objective']['model_selection']
    assert full['alignment_objective']['reverse_term']==baseline['alignment_objective']['reverse_term']
    # A completed resume with a different objective must fail, not silently mix losses.
    with pytest.raises(ValueError,match='Resume metadata mismatch'):
        train_neural(records,dict(cfg,alignment_candidate_scope='batch'),'alignment',tmp_path/'training_unique',
                     text_cache.path,resume=tmp_path/'training_unique/last.pt',control=control)


def test_validation_and_test_only_texts_never_enter_training_bank(records,config,tmp_path):
    from dataclasses import replace
    from eeg_decoding.workflow import make_splits
    from eeg_decoding.training.alignment_train import TextCache
    import eeg_decoding.training.alignment_train as alignment
    splits=make_splits(records,config)
    training_subjects={r.subject for r in splits['train']}
    changed=[r if r.subject in training_subjects else replace(r,text='HELD OUT '+r.text) for r in records]
    keys=sorted({r.key for r in changed})
    cache_path=tmp_path/'SYNTHETIC_text.npz'
    np.savez_compressed(cache_path,keys=np.array(keys),embeddings=np.random.default_rng(7).normal(size=(len(keys),12)).astype(np.float32),metadata=json.dumps({'synthetic':True}))
    cache=TextCache(cache_path)
    calls=[]; original=alignment.alignment_loss
    def spy(*args,**kwargs):
        calls.append(kwargs.get('candidate_keys'))
        return original(*args,**kwargs)
    from unittest.mock import patch
    with patch.object(alignment,'alignment_loss',spy):
        train_neural(changed,dict(config,epochs=1,alignment_candidate_scope='training_unique'),'alignment',tmp_path/'model',cache.path)
    assert any(keys is None for keys in calls)  # common batch validation, including unseen texts
    banks=[set(keys) for keys in calls if keys is not None]
    assert banks and all(bank=={r.key for r in splits['train']} for bank in banks)
    assert all(len(bank)==10 for bank in banks) and len(cache.keys)==20
