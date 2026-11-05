"""Validation-only ridge selection and controls for the fixed TRT pilot."""
import numpy as np
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler, RobustScaler
from .retrieval import ranks, summarize


def fit_word_moments(words):
    count=np.zeros(840,dtype=np.int64);total=np.zeros(840);squares=np.zeros(840)
    for a in words:
        valid=np.isfinite(a);count+=valid.sum(0)
        total+=np.where(valid,a,0).sum(0);squares+=np.where(valid,a*a,0).sum(0)
    mean=np.divide(total,count,out=np.zeros(840),where=count>0)
    variance=np.maximum(np.divide(squares,count,out=np.zeros(840),where=count>0)-mean*mean,0)
    return {'count':count,'mean':mean,'sd':np.sqrt(variance)}


def matched_gaussian(words,moments,seed):
    result=np.random.default_rng(seed).normal(moments['mean'],moments['sd'],size=words.shape)
    result[~np.isfinite(words)]=np.nan
    assert result.shape==words.shape and np.array_equal(np.isfinite(result),np.isfinite(words))
    return result


def count_basis(train_counts,counts):
    levels=np.unique(train_counts).astype(int)
    output=np.zeros((len(counts),len(levels)+1),dtype=np.float64)
    lookup={int(v):i for i,v in enumerate(levels)}
    for i,value in enumerate(counts):output[i,lookup.get(int(value),len(levels))]=1
    return output,levels


def fit_transform(train,val,imputation,scaling):
    if train.ndim!=2 or val.ndim!=2 or train.shape[1]!=val.shape[1]:raise ValueError('Invalid matrices')
    if imputation not in ('mean','median'):raise ValueError('Unknown imputation')
    fill=np.zeros(train.shape[1]);all_missing=[]
    for j in range(train.shape[1]):
        finite=train[np.isfinite(train[:,j]),j]
        if not len(finite):all_missing.append(j)
        else:fill[j]=np.mean(finite) if imputation=='mean' else np.median(finite)
    a=np.where(np.isfinite(train),train,fill);b=np.where(np.isfinite(val),val,fill)
    scaler=StandardScaler() if scaling=='standard' else RobustScaler(quantile_range=(25,75)) if scaling=='robust_iqr' else None
    if scaler is None:raise ValueError('Unknown scaling')
    a=scaler.fit_transform(a);b=scaler.transform(b)
    center=scaler.mean_ if scaling=='standard' else scaler.center_
    return a,b,{'imputation':imputation,'fill':fill,'all_missing_training_columns':all_missing,
                'scaling':scaling,'center':center,'scale':scaler.scale_}


def select_ridge(train,val,targets,val_keys,candidate_keys,candidate_vectors,config):
    """Only these train/validation arrays enter selection. No test argument exists."""
    table=[];best=-np.inf;selected=None;best_prediction=None
    for imputation in config['imputation_choices']:
        for scaling in config['scaling_choices']:
            a,b,transform=fit_transform(train,val,imputation,scaling)
            xmean=a.mean(0);ymean=targets.mean(0)
            u,s,vt=svd(a-xmean,full_matrices=False);keep=s>1e-15
            uty=u[:,keep].T@(targets-ymean);projected=(b-xmean)@vt[keep].T
            for alpha in config['ridge_alphas']:
                weights=(s[keep]/(s[keep]*s[keep]+alpha))[:,None]*uty
                prediction=projected@weights+ymean
                metric=summarize(ranks(prediction,candidate_vectors,val_keys,candidate_keys))
                table.append({'imputation':imputation,'scaling':scaling,'alpha':alpha,**metric})
                if metric['mrr']>best+1e-12:
                    best=metric['mrr'];coef=(vt[keep].T@weights).T
                    selected={'alpha':alpha,'transform':transform,'coef':coef,'intercept':ymean-xmean@coef.T}
                    best_prediction=prediction.copy()
    return selected,table,best_prediction


def useful_validation(reports,minimum_gap=.002):
    by={r['condition']:r for r in reports};real=by['real_trt']
    controls=[by[k] for k in ('matched_gaussian','class_shuffled','count_duration_only','zero_missingness_only')]
    strongest=max(controls,key=lambda r:r['metrics']['mrr'])
    checks={'mrr_margin_over_strongest_control':real['metrics']['mrr']-strongest['metrics']['mrr']>=minimum_gap,
            'recall_corroboration':sum(real['metrics'][k]>max(r['metrics'][k] for r in controls) for k in ('recall@1','recall@5','recall@10'))>=2}
    for subject in real['per_subject']:
        checks[subject+'_mrr_above_all_controls']=all(real['per_subject'][subject]['mrr']>r['per_subject'][subject]['mrr'] for r in controls)
    return {'useful_validation_advantage':all(checks.values()),'strongest_control':strongest['condition'],
            'real_minus_strongest_mrr':real['metrics']['mrr']-strongest['metrics']['mrr'],
            'checks':checks,'test_evaluated':False,'interpretation':'Prespecified descriptive screen on one reused validation split; not statistical significance.'}
