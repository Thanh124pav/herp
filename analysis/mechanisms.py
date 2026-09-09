"""Independent high-K sigma oracle and actual one-step region-update diagnostics."""
import argparse
import copy
import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train import (Args, Agent, make_env, collect_fragment, concat_batches, signature, occupancy_scores,
                   empirical_fisher_diagonal, signature_parameters, evaluate_policy)
from herp.probe import probe_region
from herp.logging import CsvLogger


def correlation(x, y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return dict(rho=None, pvalue=None, n=len(x), reason='insufficient or constant observations')
    result = spearmanr(x,y)
    return dict(rho=float(result.statistic), pvalue=float(result.pvalue), n=len(x))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--kind',choices=['sigma','p','both'],default='both')
    parser.add_argument('--regions',type=int,default=50)
    parser.add_argument('--oracle-probes',type=int,default=64)
    parser.add_argument('--small-probes',type=int,default=4)
    parser.add_argument('--reference-steps',type=int,default=512)
    parser.add_argument('--region-steps',type=int,default=64)
    parser.add_argument('--region-rollouts',type=int,default=1)
    parser.add_argument('--eval-episodes',type=int,default=50)
    parser.add_argument('--step-size',type=float,default=.001)
    parser.add_argument('--seed',type=int,default=710)
    parser.add_argument('--output-dir',default='outputs/mechanisms')
    opt=parser.parse_args()
    torch.set_num_threads(1)
    random.seed(opt.seed); np.random.seed(opt.seed); torch.manual_seed(opt.seed)
    # Only load checkpoints produced locally by this repository (they include simulator snapshots).
    cp=torch.load(opt.checkpoint,map_location='cpu',weights_only=False)
    args=Args(**cp['config']); args.device='cpu'
    archive,regionizer=cp['archive'],cp['regionizer']
    env=make_env(args); ref_env=make_env(args); oracle_env=make_env(args)
    agent=Agent(archive.regions[0].centroid.numel(),env.action_space.shape[-1],args.hidden)
    agent.load_state_dict(cp['agent'])
    candidates=[r for r in archive if r.snapshots]
    rng=np.random.default_rng(opt.seed)
    candidates=[candidates[i] for i in rng.permutation(len(candidates))[:opt.regions]]
    out=Path(opt.output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'protocol.json').write_text(json.dumps(dict(**vars(opt),training_step=cp['global_env_steps'],training_config=cp['config']),indent=2))
    summary={}; total_steps=0
    if opt.kind in ('sigma','both'):
        rows=[]; paths={}
        logger=CsvLogger(out/'sigma.csv')
        for region in candidates:
            snap=archive.sample_snapshot(region)
            kwargs=dict(num_env_repeats=args.num_env_repeats,probe_horizon=args.probe_horizon,
                        probe_scale=args.probe_scale,gamma_branch=args.gamma_branch,lambda_dyn=args.lambda_dyn,
                        estimator=args.sigma_estimator,gamma=args.gamma)
            # Independent futures; never compare a subset against an oracle containing that subset.
            small=probe_region(env,snap,agent,regionizer,num_action_probes=opt.small_probes,**kwargs)
            oracle=probe_region(env,snap,agent,regionizer,num_action_probes=opt.oracle_probes,**kwargs)
            total_steps+=small['steps']+oracle['steps']
            row=dict(region_id=region.region_id,sigma=small['sigma'],oracle_sigma=oracle['sigma'],
                     small_steps=small['steps'],oracle_steps=oracle['steps'])
            logger.log(row); rows.append(row)
            paths[region.region_id]=oracle['trajectories']
            print(json.dumps(row),flush=True)
        summary['sigma']=correlation([r['sigma'] for r in rows],[r['oracle_sigma'] for r in rows])
        torch.save(paths,out/'sigma_trajectories.pt')
        fig,ax=plt.subplots(figsize=(4,3));ax.scatter([r['sigma'] for r in rows],[r['oracle_sigma'] for r in rows])
        ax.set(xlabel=f'Sigma (K={opt.small_probes})',ylabel=f'Independent oracle (K={opt.oracle_probes})',
               title=f"Spearman rho = {summary['sigma']['rho']}")
        fig.tight_layout();fig.savefig(out/'sigma_correlation.pdf');fig.savefig(out/'sigma_correlation.png',dpi=180);plt.close(fig)
        # Plot actual future paths in a common two-dimensional state PCA basis.
        selected=sorted(rows,key=lambda r:r['oracle_sigma'])
        selected=selected[:min(5,len(rows)//2)]+selected[-min(5,len(rows)//2):]
        if selected:
            all_states=torch.cat([p for row in selected for p in paths[row['region_id']]])
            mean=all_states.mean(0);_,_,v=torch.pca_lowrank(all_states-mean,q=2)
            fig,axes=plt.subplots(2,(len(selected)+1)//2,figsize=(12,5),squeeze=False)
            for ax,row in zip(axes.flat,selected):
                for path in paths[row['region_id']][:16]:
                    z=(path-mean)@v;ax.plot(z[:,0],z[:,1],alpha=.3,lw=.7)
                ax.set_title(f"Region {row['region_id']}; sigma={row['oracle_sigma']:.2f}")
            fig.suptitle('Low/high dispersion: future trajectories in shared state PCA coordinates')
            fig.tight_layout();fig.savefig(out/'sigma_futures.pdf');plt.close(fig)
    if opt.kind in ('p','both'):
        reference=collect_fragment(ref_env,agent,args,opt.reference_steps,3,reset_seed=3_000_000+opt.seed)
        heldout=collect_fragment(ref_env,agent,args,opt.reference_steps,3,reset_seed=3_100_000+opt.seed)
        total_steps+=reference['steps']+heldout['steps']
        g_ref=signature(agent,reference,'cpu')
        fisher=empirical_fisher_diagonal(agent,reference['obs'],reference['actions'])
        occ=occupancy_scores(regionizer,reference['obs'])
        baseline=evaluate_policy(oracle_env,agent,opt.eval_episodes,'cpu',seed_base=3_200_000+opt.seed)
        total_steps+=baseline['eval_steps']
        adv=heldout['advantages'];adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
        def surrogate(policy):
            with torch.no_grad():
                logp=policy.get_distribution(heldout['obs']).log_prob(heldout['actions']).sum(-1)
                return float(((logp-heldout['logprobs']).exp()*adv).mean())
        base_surrogate=surrogate(agent)
        logger=CsvLogger(out/'p.csv');rows=[]
        for region in candidates:
            snap=archive.sample_snapshot(region)
            batch=concat_batches([collect_fragment(env,agent,args,opt.region_steps,2,start_snapshot=snap)
                                  for _ in range(opt.region_rollouts)])
            total_steps+=batch['steps']
            g=signature(agent,batch,'cpu')
            dot=float(g_ref@g)
            cosine=dot/max(float(g.norm()*g_ref.norm()),1e-12)
            natural=float(g_ref@(g/(fisher+args.fisher_damping)))
            clone=copy.deepcopy(agent)
            with torch.no_grad():
                offset=0
                for param in signature_parameters(clone):
                    param.add_(g[offset:offset+param.numel()].view_as(param),alpha=-opt.step_size)
                    offset+=param.numel()
            after=evaluate_policy(oracle_env,clone,opt.eval_episodes,'cpu',seed_base=3_200_000+opt.seed)
            total_steps+=after['eval_steps']
            row=dict(region_id=region.region_id,occupancy=float(occ[region.region_id]),cosine=cosine,dot=dot,
                     fisher=natural,gradient_norm=float(g.norm()),delta_return=after['eval_return']-baseline['eval_return'],
                     delta_success=after['eval_success']-baseline['eval_success'],
                     delta_surrogate=surrogate(clone)-base_surrogate,region_steps=batch['steps'])
            rows.append(row);logger.log(row);print(json.dumps(row),flush=True)
        summary['p']={key:{target:correlation([r[key] for r in rows],[r[target] for r in rows])
                          for target in ('delta_return','delta_surrogate')}
                      for key in ('occupancy','cosine','dot','fisher')}
        fig,axes=plt.subplots(1,4,figsize=(12,3))
        for ax,key in zip(axes,('occupancy','cosine','dot','fisher')):
            ax.scatter([r[key] for r in rows],[r['delta_return'] for r in rows])
            rho=summary['p'][key]['delta_return']['rho']
            ax.set(xlabel=key,ylabel='Paired change in return',title=f'rho={rho:.2f}' if rho is not None else 'Undefined correlation')
        fig.tight_layout();fig.savefig(out/'p_correlation.pdf');fig.savefig(out/'p_correlation.png',dpi=180);plt.close(fig)
    summary['diagnostic_env_steps']=total_steps
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    for instance in (env,ref_env,oracle_env):instance.close()


if __name__=='__main__':main()
