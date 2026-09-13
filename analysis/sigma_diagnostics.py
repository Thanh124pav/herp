"""Independent-oracle ranking, holdout ridge, and repeated-subsample bias/variance."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr,kendalltau
from sklearn.metrics import ndcg_score,r2_score
from herp.sigma_predictor import LinearVariancePredictor


def q_est(x,mask):
    if mask.all():return x.var(0,ddof=1).sum(-1).mean()
    vals=[]
    for i in range(len(x)):
        for j in range(i+1,len(x)):
            common=mask[i]&mask[j]
            if common.sum()>=max(2,x.shape[1]//4):vals.append(np.square(x[i,common]-x[j,common]).sum(-1).mean()/2)
    return np.mean(vals) if vals else np.nan


def metrics(pred,truth):
    ok=np.isfinite(pred)&np.isfinite(truth);p,t=pred[ok],truth[ok]
    if len(t)<3:return {}
    k=max(1,int(np.ceil(.2*len(t))))
    constant=np.std(p)<1e-12 or np.std(t)<1e-12
    return dict(pearson=np.nan if constant else float(pearsonr(p,t).statistic),
        spearman=np.nan if constant else float(spearmanr(p,t).statistic),
        kendall=np.nan if constant else float(kendalltau(p,t).statistic),
        top20_overlap=len(set(np.argsort(p)[-k:])&set(np.argsort(t)[-k:]))/k,
        ndcg=float(ndcg_score(t[None],p[None])),normalized_rmse=float(np.sqrt(np.mean((p-t)**2))/(np.std(t)+1e-12)),
        normalized_mae=float(np.mean(abs(p-t))/(np.mean(abs(t))+1e-12)),n=len(t))


def main():
    p=argparse.ArgumentParser();p.add_argument('--pool',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--resamples',type=int,default=100)
    a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    data=np.load(a.pool);features=data['features'].astype(np.float64);valid=data['valid'];X=data['X'];low_n=int(data['low_pool'])
    low,oracle=features[:,:low_n],features[:,low_n:];lm,om=valid[:,:low_n],valid[:,low_n:]
    qstar=np.array([q_est(x,m) for x,m in zip(oracle,om)]);sigma=np.sqrt(qstar)
    rng=np.random.default_rng(314);order=rng.permutation(len(X));cut=max(1,int(.8*len(X)));train,test=order[:cut],order[cut:]
    rows=[];bias_rows=[];predictions={};stability=[];coefficients=[]
    for n in [32,64,128]:
        if n>oracle.shape[1]:continue
        q=np.array([q_est(x[:n],m[:n]) for x,m in zip(oracle,om)])
        stability.append(dict(K=n,**metrics(np.sqrt(q),sigma)))
    for K in [2,4,8,16,32]:
        if K>low_n:continue
        samples={e:[] for e in ['direct','ridge_holdout','shrinkage_holdout','one_step','endpoint','mean_compressed','legacy_discounted']}
        for repeat in range(a.resamples):
            ix=[rng.choice(low_n,K,replace=False) for _ in X]
            selected=np.array([low[i,j] for i,j in enumerate(ix)]);masks=np.array([lm[i,j] for i,j in enumerate(ix)])
            direct=np.array([q_est(x,m) for x,m in zip(selected,masks)])
            model=LinearVariancePredictor(5,1e-3)
            for i in train:model.add_label(X[i],direct[i],K)
            model.fit();pred=np.array([model.predict(x) for x in X])
            coefficients.append(dict(K=K,repeat=repeat,coefficients=model.raw_coefficients().tolist()))
            shrink=K/(K+8)*direct+8/(K+8)*pred
            one=selected[:,:,0].var(1,ddof=1).sum(-1)
            endpoint=np.full(len(X),np.nan);mean=endpoint.copy();legacy=endpoint.copy()
            weights=.9**np.arange(selected.shape[2]);weights/=weights.sum()
            for i in range(len(X)):
                complete=selected[i,masks[i].all(-1)]
                if len(complete)<2:continue
                endpoint[i]=complete[:,-1].var(0,ddof=1).sum()
                mean[i]=complete.mean(1).var(0,ddof=1).sum()
                legacy[i]=(complete*weights[None,:,None]).sum(1).var(0,ddof=1).sum()
            for name,values in zip(samples,[direct,pred,shrink,one,endpoint,mean,legacy]):
                samples[name].append(values)
                scope=test if 'holdout' in name else np.arange(len(X))
                rows.append(dict(K=K,repeat=repeat,estimator=name,scope='heldout_regions' if 'holdout' in name else 'all_regions',
                                 **metrics(np.sqrt(np.maximum(0,values[scope])),sigma[scope])))
                # Direct on exactly the same held-out regions for fair comparisons.
                if name=='direct':rows.append(dict(K=K,repeat=repeat,estimator='direct_holdout',scope='heldout_regions',**metrics(np.sqrt(values[test]),sigma[test])))
        for name,vals in samples.items():
            vals=np.array(vals);scope=test if 'holdout' in name else np.arange(len(X))
            bias2=np.square(vals[:,scope].mean(0)-qstar[scope]);var=vals[:,scope].var(0,ddof=0)
            bias_rows.append(dict(K=K,estimator=name,bias_squared=float(bias2.mean()),variance=float(var.mean()),mse=float((bias2+var).mean()),n_regions=len(scope)))
            if name=='direct':
                b=np.square(vals[:,test].mean(0)-qstar[test]);v=vals[:,test].var(0)
                bias_rows.append(dict(K=K,estimator='direct_holdout',bias_squared=float(b.mean()),variance=float(v.mean()),mse=float((b+v).mean()),n_regions=len(test)))
            predictions[f'{name}_K{K}']=vals
    pd.DataFrame(rows).to_csv(out/'ranking_resamples.csv',index=False)
    pd.DataFrame(bias_rows).to_csv(out/'bias_variance.csv',index=False)
    pd.DataFrame(stability).to_csv(out/'oracle_stability.csv',index=False)
    np.savez_compressed(out/'predictions.npz',qstar=qstar,sigma=sigma,heldout=test,train=train,**predictions)
    (out/'coefficients.json').write_text(json.dumps(coefficients))
    (out/'protocol.json').write_text(json.dumps(dict(region_holdout=test.tolist(),train_regions=train.tolist(),
        resamples=a.resamples,oracle_pool_independent=True,notes=[
        'Intervals over low-pool resampling describe estimator sampling variability, not training-seed uncertainty.',
        'Linear predictors refit only on training regions; direct_holdout uses the identical held-out set.',
        'Endpoint and legacy_discounted are fixed-window proxies; they are not full variable-chain-length ablations.',
        'Early termination may make masked pairwise estimates conditional on common survival.']),indent=2))
    print(pd.DataFrame(rows).groupby(['K','estimator']).spearman.mean().to_string())

if __name__=='__main__':main()
