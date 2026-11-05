import numpy as np
import pytest
from sklearn.linear_model import Ridge
from eeg_decoding.evaluation.incremental_ridge import select_structural,select_combined,ridge_path_predictions
from eeg_decoding.evaluation.spectral_pilot import fit_transform


def fixture():
    rng=np.random.default_rng(71)
    return rng.normal(size=(18,3)),rng.normal(size=(5,3)),rng.normal(size=(18,4)),rng.normal(size=(5,4)),rng.normal(size=(18,6)),rng.normal(size=(3,6))


def test_zero_weight_exactly_reproduces_structural_and_ignores_eeg():
    s,v,e,f,y,pool=fixture();cfg={'imputation_choices':['mean'],'scaling_choices':['standard'],'ridge_alphas':[.1,10.],'eeg_block_weights':[0]}
    chosen,_,prediction,a,b=select_structural(s,v,y,['a','b','c','a','b'],['a','b','c'],pool,cfg)
    left,table,pred=select_combined(a,b,e,f,y,['a','b','c','a','b'],['a','b','c'],pool,cfg)
    right,_,pred2=select_combined(a,b,e*1e12,f-1e20,y,['a','b','c','a','b'],['a','b','c'],pool,cfg)
    np.testing.assert_array_equal(pred,prediction);np.testing.assert_array_equal(pred2,prediction)
    assert left['alpha']==chosen['alpha'] and not left['coef'][:,3:].any()
    np.testing.assert_array_equal(a,fit_transform(s,v,'mean','standard')[0])


def test_weight_is_applied_after_scaling_and_matches_reference_ridge():
    s,v,e,f,y,_=fixture();a,b,_=fit_transform(s,v,'median','standard');c,d,_=fit_transform(e,f,'mean','standard')
    for weight in [.03,1.,3.]:
        x=np.c_[a,weight*c];z=np.c_[b,weight*d]
        _,pred,fit=next(ridge_path_predictions(x,z,y,[2.]))
        np.testing.assert_allclose(pred,Ridge(alpha=2.,solver='svd').fit(x,y).predict(z),atol=1e-12)
    small=next(ridge_path_predictions(np.c_[a,.03*c],np.c_[b,.03*d],y,[2.]))[1]
    large=next(ridge_path_predictions(np.c_[a,3*c],np.c_[b,3*d],y,[2.]))[1]
    assert not np.allclose(small,large)


def test_both_blocks_fit_preprocessing_from_training_only():
    s,v,e,f,_,_=fixture();s[0,0]=np.nan;e[0,0]=np.nan
    for block,held in [(s,v),(e,f)]:
        _,_,left=fit_transform(block,held,'median','robust_iqr')
        _,_,right=fit_transform(block,held*1e20,'median','robust_iqr')
        for k in ['fill','center','scale']:np.testing.assert_array_equal(left[k],right[k])


def test_grid_has_same_candidates_and_zero_per_preprocessor():
    s,v,e,f,y,p=fixture();cfg={'imputation_choices':['mean','median'],'scaling_choices':['standard','robust_iqr'],'ridge_alphas':[.1,10.],'eeg_block_weights':[0,.1]}
    _,table,_=select_combined(s,v,e,f,y,['a','b','c','a','b'],['a','b','c'],p,cfg)
    assert len(table)==16 and sum(t['weight']==0 for t in table)==8
    with pytest.raises(ValueError):select_combined(s,v,e,f,y,['a','b','c','a','b'],['a','b','c'],p,dict(cfg,eeg_block_weights=[.1]))
