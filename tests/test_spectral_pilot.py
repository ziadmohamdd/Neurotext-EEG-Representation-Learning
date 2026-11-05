import numpy as np
from scipy.io import savemat
from sklearn.linear_model import Ridge
from eeg_decoding.data.trt_view import iter_trt_sentences,mean_features,structure,count_duration,BANDS
from eeg_decoding.evaluation.spectral_pilot import fit_word_moments,matched_gaussian,count_basis,fit_transform,select_ridge,useful_validation


def test_trt_reader_excludes_word_text_and_preserves_slot_indices(tmp_path):
    def save(path,secret):
        word={'content':secret,'TRT':12,'nFixations':1,**{'TRT_'+b:np.ones((1,105))*(i+1) for i,b in enumerate(BANDS)}}
        # Non-numeric data in an unselected field must be skipped, not embedded.
        word['rawET']={'identity':'forbidden'}
        savemat(path,{'sentenceData':np.array([{'content':'fixed target','word':word},{'content':'another target','word':word}],dtype=object)},do_compression=True)
    # Actual exports use struct arrays, not the cell fixture above.
    dtype=np.dtype([('content',object),('word',object)])
    for i,secret in enumerate(['secret A','different secret B']):
        a=np.empty((1,2),dtype=dtype)
        for slot in range(2):
            a['content'][0,slot]='fixed target'
            a['word'][0,slot]={'content':secret,'rawET':{'identity':'forbidden'},'TRT':12,'nFixations':1,
                **{'TRT_'+b:np.ones((1,105))*(j+1) for j,b in enumerate(BANDS)}}
        savemat(tmp_path/f'{i}.mat',{'sentenceData':a},do_compression=True)
    left=list(iter_trt_sentences(tmp_path/'0.mat'));right=list(iter_trt_sentences(tmp_path/'1.mat'))
    assert [x[0] for x in left]==[0,1]
    assert left[0][2].shape==(1,840)
    np.testing.assert_array_equal(left[0][2],right[0][2])
    np.testing.assert_array_equal(left[0][2][0,:105],1)
    np.testing.assert_array_equal(left[0][3],[12])


def test_aggregation_and_all_missing_query_are_retained():
    words=np.stack([np.ones(840),np.full(840,np.nan),np.full(840,3.)])
    np.testing.assert_array_equal(mean_features(words),2.)
    np.testing.assert_allclose(structure(words),[3,2,2/3])
    assert np.isnan(mean_features(np.full((2,840),np.nan))).all()
    assert mean_features(np.empty((0,840))).shape==(840,)
    assert count_duration(words,np.array([4,np.nan,8]),np.array([1,np.nan,2])).shape==(14,)


def test_gaussian_preserves_every_word_and_coordinate_mask():
    train=np.stack([np.zeros(840),np.ones(840)*2])
    moments=fit_word_moments([train])
    val=np.ones((5,840))*1e9;val[1,:]=np.nan;val[2,10]=np.nan
    result=matched_gaussian(val,moments,42)
    np.testing.assert_array_equal(np.isfinite(result),np.isfinite(val))
    assert result.shape==val.shape
    np.testing.assert_array_equal(moments['mean'],1.)
    np.testing.assert_array_equal(moments['sd'],1.)
    assert np.nanmax(np.abs(result))<10
    np.testing.assert_array_equal(result,matched_gaussian(val,moments,42))


def test_count_basis_uses_training_values_and_an_unknown_bucket():
    a,levels=count_basis(np.array([5,5,9]),np.array([5,7,9]))
    assert levels.tolist()==[5,9]
    np.testing.assert_array_equal(a,[[1,0,0],[0,0,1],[0,1,0]])


def test_imputer_and_scaler_are_training_only():
    train=np.array([[1,np.nan,np.nan],[3,8,np.nan],[5,12,np.nan]],float)
    val=np.array([[1e9,-1e9,7]],float)
    for imputation in ('mean','median'):
        for scaling in ('standard','robust_iqr'):
            a,b,fit=fit_transform(train,val,imputation,scaling)
            _,_,other=fit_transform(train,np.zeros_like(val),imputation,scaling)
            np.testing.assert_array_equal(fit['fill'],[3,10,0])
            np.testing.assert_array_equal(fit['center'],other['center'])
            np.testing.assert_array_equal(fit['scale'],other['scale'])
            assert fit['all_missing_training_columns']==[2] and np.isfinite(a).all() and np.isfinite(b).all()


def test_ridge_prediction_matches_sklearn_svd():
    rng=np.random.default_rng(8);train=rng.normal(size=(20,4));val=rng.normal(size=(5,4));y=rng.normal(size=(20,6));candidates=rng.normal(size=(3,6))
    cfg={'imputation_choices':['mean'],'scaling_choices':['standard'],'ridge_alphas':[.1,10.]}
    chosen,table,pred=select_ridge(train,val,y,['a','b','c','a','b'],['a','b','c'],candidates,cfg)
    a,b,_=fit_transform(train,val,'mean','standard')
    reference=Ridge(alpha=chosen['alpha'],solver='svd').fit(a,y).predict(b)
    np.testing.assert_allclose(pred,reference,atol=1e-12)
    assert len(table)==2 and table[0]['n_queries']==5


def test_no_useful_gate_when_a_structural_control_is_stronger():
    reports=[]
    for name in ('real_trt','matched_gaussian','class_shuffled','count_duration_only','zero_missingness_only'):
        v=.2 if name=='count_duration_only' else .1
        m={'mrr':v,'recall@1':v,'recall@5':v,'recall@10':v}
        reports.append({'condition':name,'metrics':m,'per_subject':{'reader':m}})
    decision=useful_validation(reports)
    assert not decision['useful_validation_advantage'] and not decision['test_evaluated']
    assert decision['strongest_control']=='count_duration_only'
