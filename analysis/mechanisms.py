"""Independent high-K sigma oracle and controlled PPO-delta region diagnostics.

The p mechanism test follows IMPL §25 controlled PPO-delta rather than the older
one-step SGD proxy:

    theta_A = PPOUpdate(theta, D_base)
    theta_B = PPOUpdate(theta, D_base U D_v)
    Delta_v = J_ref(theta_B) - J_ref(theta_A)

Both clones share the same starting checkpoint, the same PPO hyperparameters and
the same optimizer state, so Delta_v isolates the marginal contribution of
adding region v's data on top of an ordinary PPO update.
"""
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
from train import (Args, Agent, build_adapter, collect_fragment, concat_batches, signature, occupancy_scores,
                   empirical_fisher_diagonal, evaluate_policy, ppo_update)
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
    # Retained for backward compatibility with older logs; the controlled PPO-delta
    # test does not use a raw SGD step, so --step-size is now ignored.
    parser.add_argument('--step-size',type=float,default=.001,
                        help='Deprecated: unused by controlled PPO-delta.')
    parser.add_argument('--seed',type=int,default=710)
    parser.add_argument('--output-dir',default='outputs/mechanisms')
    opt=parser.parse_args()
    torch.set_num_threads(1)
    random.seed(opt.seed); np.random.seed(opt.seed); torch.manual_seed(opt.seed)
    # Only load checkpoints produced locally by this repository (they include simulator snapshots).
    cp=torch.load(opt.checkpoint,map_location='cpu',weights_only=False)
    args=Args(**cp['config']); args.device='cpu'
    archive,regionizer=cp['archive'],cp['regionizer']
    adapter=build_adapter(args); ref_adapter=build_adapter(args); oracle_adapter=build_adapter(args)
    env,ref_env,oracle_env=adapter.env,ref_adapter.env,oracle_adapter.env
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
                        estimator=args.sigma_estimator,gamma=args.gamma,adapter=adapter)
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
        # D_base -- an ordinary PPO batch that both clones will train on.
        # A held-out fragment stays around only for the Taylor surrogate check.
        reference=collect_fragment(ref_env,agent,args,opt.reference_steps,3,reset_seed=3_000_000+opt.seed,adapter=ref_adapter)
        heldout=collect_fragment(ref_env,agent,args,opt.reference_steps,3,reset_seed=3_100_000+opt.seed,adapter=ref_adapter)
        total_steps+=reference['steps']+heldout['steps']
        adv_scale=max(float(reference['advantages'].std(unbiased=False)),1e-6)
        g_ref=signature(agent,reference,'cpu',adv_scale=adv_scale)
        fisher=empirical_fisher_diagonal(agent,reference['obs'],reference['actions'])
        occ=occupancy_scores(regionizer,reference['obs'])
        optimizer_state=cp.get('optimizer')  # share exact optimizer state across clones
        adv=heldout['advantages']  # raw, matches signature convention
        def surrogate(policy):
            with torch.no_grad():
                logp=policy.get_distribution(heldout['obs']).log_prob(heldout['actions']).sum(-1)
                return float(((logp-heldout['logprobs']).exp()*adv).mean())
        base_surrogate=surrogate(agent)

        def ppo_clone(base_batch,extra_batch,ppo_seed):
            """Return (theta', eval_dict) after one PPO update on base_batch (+ optional extra)."""
            clone=copy.deepcopy(agent)
            optimizer=torch.optim.Adam(clone.parameters(),lr=args.learning_rate,eps=1e-5)
            if optimizer_state is not None:
                try:
                    optimizer.load_state_dict(copy.deepcopy(optimizer_state))
                except Exception:
                    pass
            batch=base_batch if extra_batch is None else concat_batches([base_batch,extra_batch])
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(ppo_seed)
                ppo_metrics=ppo_update(clone,optimizer,batch,args)
            evald=evaluate_policy(oracle_env,clone,opt.eval_episodes,'cpu',
                                  seed_base=3_200_000+opt.seed,adapter=oracle_adapter)
            return clone,ppo_metrics,evald

        # Clone A -- ordinary PPO update on D_base (same for every region).
        clone_A,ppo_A_metrics,eval_A=ppo_clone(reference,None,ppo_seed=opt.seed+7919)
        surrogate_A=surrogate(clone_A)
        total_steps+=eval_A['eval_steps']

        logger=CsvLogger(out/'p.csv');rows=[]
        eta=getattr(args,'hybrid_eta',0.5); eps_h=getattr(args,'eps_hybrid',1e-6)
        for region in candidates:
            snap=archive.sample_snapshot(region)
            batch=concat_batches([collect_fragment(env,agent,args,opt.region_steps,2,start_snapshot=snap,adapter=adapter)
                                  for _ in range(opt.region_rollouts)])
            total_steps+=batch['steps']
            g=signature(agent,batch,'cpu',adv_scale=adv_scale)
            dot=float(g_ref@g)
            cosine=dot/max(float(g.norm()*g_ref.norm()),1e-12)
            natural=float(g_ref@(g/(fisher+args.fisher_damping)))
            occ_val=float(occ[region.region_id])
            hybrid=(occ_val+eps_h)**eta*(max(0.,cosine)+eps_h)**(1.-eta)
            # Clone B -- PPO update on D_base U D_v with matched hyperparameters.
            clone_B,ppo_B_metrics,eval_B=ppo_clone(reference,batch,ppo_seed=opt.seed+7919)
            surrogate_B=surrogate(clone_B)
            total_steps+=eval_B['eval_steps']
            row=dict(region_id=region.region_id,occupancy=occ_val,cosine=cosine,dot=dot,
                     fisher=natural,hybrid=hybrid,gradient_norm=float(g.norm()),
                     ppo_delta_return=eval_B['eval_return']-eval_A['eval_return'],
                     ppo_delta_success=eval_B['eval_success']-eval_A['eval_success'],
                     ppo_delta_surrogate=surrogate_B-surrogate_A,
                     eval_A_return=eval_A['eval_return'],eval_B_return=eval_B['eval_return'],
                     eval_A_success=eval_A['eval_success'],eval_B_success=eval_B['eval_success'],
                     region_steps=batch['steps'])
            rows.append(row);logger.log(row);print(json.dumps(row),flush=True)
        summary['p']={key:{target:correlation([r[key] for r in rows],[r[target] for r in rows])
                          for target in ('ppo_delta_return','ppo_delta_success','ppo_delta_surrogate')}
                      for key in ('occupancy','cosine','dot','fisher','hybrid')}
        summary['p_reference']={'eval_A_return':eval_A['eval_return'],
                                'eval_A_success':eval_A['eval_success'],
                                'ppo_metrics_A':ppo_A_metrics}
        fig,axes=plt.subplots(1,5,figsize=(15,3))
        for ax,key in zip(axes,('occupancy','cosine','dot','fisher','hybrid')):
            ax.scatter([r[key] for r in rows],[r['ppo_delta_return'] for r in rows])
            rho=summary['p'][key]['ppo_delta_return']['rho']
            ax.set(xlabel=key,ylabel='PPO delta return (J_ref(B) - J_ref(A))',
                   title=f'rho={rho:.2f}' if rho is not None else 'Undefined correlation')
        fig.tight_layout();fig.savefig(out/'p_correlation.pdf');fig.savefig(out/'p_correlation.png',dpi=180);plt.close(fig)
    summary['diagnostic_env_steps']=total_steps
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    for a in (adapter,ref_adapter,oracle_adapter):a.close()


if __name__=='__main__':main()
