"""Paper-sized mechanism figures and region-bootstrap uncertainty from raw CSV."""
import argparse
import csv
import json
from pathlib import Path
import warnings
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def stats(x,y,seed=391,repeats=2000):
    x,y=np.asarray(x),np.asarray(y)
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:return dict(rho=None,ci95=None,n=len(x))
    rho=float(spearmanr(x,y).statistic);rng=np.random.default_rng(seed);draws=[]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for _ in range(repeats):
            idx=rng.integers(0,len(x),len(x));r=spearmanr(x[idx],y[idx]).statistic
            if np.isfinite(r):draws.append(float(r))
    return dict(rho=rho,ci95=np.quantile(draws,[.025,.975]).tolist() if draws else None,n=len(x),bootstrap_valid=len(draws))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output-dir',required=True)
    args=p.parse_args();root=Path(args.root);out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    sigma_files=[f for f in sorted(root.glob('sigma_*/sigma.csv'),key=lambda f:int(f.parent.name.split('_')[-1])) if (f.parent/'summary.json').exists()]
    summary={}
    if sigma_files:
        fig,axes=plt.subplots(1,len(sigma_files),figsize=(3.35*len(sigma_files),2.8),squeeze=False)
        for ax,path in zip(axes.flat,sigma_files):
            rows=read(path);x=[float(r['sigma']) for r in rows];y=[float(r['oracle_sigma']) for r in rows]
            result=stats(x,y);summary[path.parent.name]=result
            protocol=json.loads((path.parent/'protocol.json').read_text())
            ax.scatter(x,y,s=17,color='#235d8d',alpha=.85,edgecolors='none')
            ax.set(xlabel='Trajectory dispersion (K=4)',ylabel='Independent oracle (K=64)',
                   title=f"{protocol['training_step']:,} interactions")
            ax.text(.04,.95,f"Spearman ρ = {result['rho']:.3f}\nn = {len(rows)} regions",transform=ax.transAxes,va='top',fontsize=8)
        fig.tight_layout();fig.savefig(out/'sigma_validation.pdf',bbox_inches='tight');fig.savefig(out/'sigma_validation.png',dpi=220,bbox_inches='tight');plt.close(fig)
    for path in sorted(root.glob('p_*/p.csv')):
        if not (path.parent/'summary.json').exists():continue
        rows=read(path)
        # Controlled PPO-delta is the canonical target; fall back to the legacy
        # one-step-SGD column when reading older CSVs.
        target_key='ppo_delta_return' if rows and 'ppo_delta_return' in rows[0] else 'delta_return'
        y_label=('J_ref(theta_B) - J_ref(theta_A)' if target_key=='ppo_delta_return'
                 else 'Paired change in return')
        y=[float(r[target_key]) for r in rows]
        fig,axes=plt.subplots(1,4,figsize=(7,2.5));summary[path.parent.name]={}
        for ax,key,label in zip(axes,('occupancy','cosine','dot','fisher'),('Occupancy','Cosine','Dot product','Diagonal Fisher')):
            x=[float(r[key]) for r in rows];result=stats(x,y);summary[path.parent.name][key]=result
            ax.scatter(x,y,s=12,color='#a64b35',alpha=.8,edgecolors='none');ax.axhline(0,color='.6',lw=.6)
            ax.set_xlabel(label)
            ax.set_title(f"ρ = {result['rho']:.3f}" if result['rho'] is not None else 'ρ undefined')
        axes[0].set_ylabel(y_label)
        fig.tight_layout();fig.savefig(out/'p_validation.pdf',bbox_inches='tight');fig.savefig(out/'p_validation.png',dpi=220,bbox_inches='tight');plt.close(fig)
    (out/'correlations_bootstrap.json').write_text(json.dumps(summary,indent=2))
    lines=['# Mechanism correlations','', 'Bootstrap percentile intervals resample archived regions (2,000 resamples). They quantify variation across this checkpoint’s sampled regions, not across training seeds.','',
           '| Diagnostic | Estimator | Spearman rho | 95% bootstrap interval | Regions |','|---|---|---:|---|---:|']
    for name,value in summary.items():
        entries={'K4 vs independent K64':value} if name.startswith('sigma') else value
        for estimator,result in entries.items():
            ci=result['ci95'];interval=f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else 'undefined'
            rho=f"{result['rho']:.3f}" if result['rho'] is not None else 'undefined'
            lines.append(f"| {name} | {estimator} | {rho} | {interval} | {result['n']} |")
    (out/'MECHANISMS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
