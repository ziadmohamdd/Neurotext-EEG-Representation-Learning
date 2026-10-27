import numpy as np
import pytest
from eeg_decoding.evaluation.retrieval import ranks,summarize,chance
from eeg_decoding.evaluation.controls import gaussian_input
from eeg_decoding.evaluation.robustness import corrupt
from eeg_decoding.evaluation.statistics import subject_bootstrap

def test_known_retrieval():
    identity=np.eye(6,dtype=np.float32)
    keys=list("abcdef")
    assert summarize(ranks(identity,identity,keys,keys))["top1"]==1
    assert summarize(ranks(identity,identity,keys[::-1],keys))["top1"]==0
    assert chance(6)["top5"]==5/6
    with pytest.raises(ValueError):
        ranks(identity,identity,keys,["a"]*6)
    with pytest.raises(ValueError):
        ranks(identity,identity,["missing"]*6,keys)

def test_ties_not_optimistic():
    rank=ranks(np.ones((2,3)),np.ones((4,3)),["a","b"],list("abcd"))
    np.testing.assert_array_equal(rank,[2.5,2.5])
    assert summarize(rank)["top1"]==0

@pytest.mark.parametrize("kind",["additive_noise","channel_dropout","temporal_mask"])
def test_robustness(kind):
    x=np.random.default_rng(0).normal(size=(8,1000)).astype(np.float32)
    np.testing.assert_array_equal(corrupt(x,kind,0,np.random.default_rng(1)),x)
    changed=corrupt(x,kind,.5,np.random.default_rng(1))
    assert changed.shape==x.shape and not np.array_equal(changed,x)
    np.testing.assert_array_equal(x,np.random.default_rng(0).normal(size=(8,1000)).astype(np.float32))

def test_noise_control():
    x=np.random.default_rng(0).normal(size=(3,10000)).astype(np.float32)
    noise=gaussian_input(x,np.random.default_rng(2))
    assert abs(np.corrcoef(x.ravel(),noise.ravel())[0,1])<.03
    np.testing.assert_allclose(noise.std(-1),x.std(-1),rtol=.03)

def test_cluster_bootstrap():
    result=subject_bootstrap([0,1,1,1],["a","a","b","b"])
    assert result["mean"]==.75 and result["ci95"][0]<=.75<=result["ci95"][1]
    assert subject_bootstrap([1],["a"])["ci95"] is None
