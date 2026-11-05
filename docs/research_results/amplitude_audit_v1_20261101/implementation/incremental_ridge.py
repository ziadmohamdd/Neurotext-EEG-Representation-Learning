"""Ridge block weighting with an exact structural-only nested candidate."""
import numpy as np
from scipy.linalg import svd
from .retrieval import ranks,summarize
from .spectral_pilot import fit_transform,select_ridge


def ridge_path_predictions(train,val,targets,alphas):
    xmean=train.mean(0);ymean=targets.mean(0)
    u,s,vt=svd(train-xmean,full_matrices=False);keep=s>1e-15
    uty=u[:,keep].T@(targets-ymean);projected=(val-xmean)@vt[keep].T
    for alpha in alphas:
        weights=(s[keep]/(s[keep]*s[keep]+alpha))[:,None]*uty
        coef=(vt[keep].T@weights).T
        yield alpha,projected@weights+ymean,{'coef':coef,'intercept':ymean-xmean@coef.T}


def select_structural(train,val,targets,val_keys,pool,vectors,config):
    selected,table,prediction=select_ridge(train,val,targets,val_keys,pool,vectors,config)
    transform=selected['transform']
    a,b,fit=fit_transform(train,val,transform['imputation'],transform['scaling'])
    for key in ('fill','center','scale'):np.testing.assert_array_equal(fit[key],transform[key])
    return selected,table,prediction,a,b


def select_combined(structural_train,structural_val,eeg_train,eeg_val,targets,val_keys,pool,vectors,config):
    if len(structural_train)!=len(eeg_train) or len(structural_val)!=len(eeg_val):raise ValueError('Block rows differ')
    weights=config['eeg_block_weights']
    if not weights or weights[0]!=0 or weights!=sorted(set(weights)) or any(not np.isfinite(w) or w<0 for w in weights):
        raise ValueError('Declare ascending finite nonnegative EEG weights, starting at zero')
    base=list(ridge_path_predictions(structural_train,structural_val,targets,config['ridge_alphas']))
    transformed={}
    for imputation in config['imputation_choices']:
        for scaling in config['scaling_choices']:
            transformed[imputation,scaling]=fit_transform(eeg_train,eeg_val,imputation,scaling)
    best=-np.inf;choice=None;best_prediction=None;table=[]
    for weight in weights:
        for imputation in config['imputation_choices']:
            for scaling in config['scaling_choices']:
                a,b,transform=transformed[imputation,scaling]
                # Weight after all train-fitted scaling. Never fit a scaler on
                # the concatenated weighted blocks: it would cancel the weight.
                path=base if weight==0 else ridge_path_predictions(
                    np.c_[structural_train,weight*a],np.c_[structural_val,weight*b],targets,config['ridge_alphas'])
                for alpha,prediction,fit in path:
                    metric=summarize(ranks(prediction,vectors,val_keys,pool))
                    table.append({'weight':weight,'eeg_imputation':imputation,'eeg_scaling':scaling,'alpha':alpha,**metric})
                    if metric['mrr']>best+1e-12:
                        best=metric['mrr'];coef=fit['coef']
                        if weight==0:coef=np.c_[coef,np.zeros((coef.shape[0],eeg_train.shape[1]))]
                        choice={'weight':weight,'alpha':alpha,'eeg_transform':transform,
                                'coef':coef.copy(),'intercept':fit['intercept'].copy()}
                        best_prediction=prediction.copy()
    return choice,table,best_prediction
