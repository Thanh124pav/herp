"""Frozen-policy, independent-pool sigma oracle; diagnostic interactions only."""
import argparse,json,sys,time
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'src')]
import numpy as np
import torch
from train import Agent
from herp.envs import make_adapter
from herp.acquisition import FragmentCollector,AcquisitionJob
from herp.vector_acquisition import VectorFragmentCollector
from herp.sigma_predictor import region_predictor_features

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',required=True);p.add_argument('--output-dir',required=True)
p.add_argument('--benchmark',default='');p.add_argument('--env-id',default='')
p.add_argument('--control-mode',default='pd_ee_delta_pose');p.add_argument('--reward-mode',default='dense')
p.add_argument('--fragments-per-region',type=int,default=128)
p.add_argument('--regions',type=int,default=30);p.add_argument('--oracle-samples',type=int,default=128)
p.add_argument('--low-pool',type=int,default=32);p.add_argument('--seed',type=int,default=912)
p.add_argument('--wandb-mode',choices=['disabled','offline','online'],default='online')
p.add_argument('--wandb-project',default='');p.add_argument('--wandb-run-name',default='')
a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(a.seed);np.random.seed(a.seed)
s=torch.load(a.checkpoint,weights_only=False,map_location='cpu');cfg=s['cfg']
out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
bench=a.benchmark or s['args'].get('benchmark','maniskill')
env_id=a.env_id or s['args']['env_id']
run=None
if a.wandb_mode!='disabled':
    import wandb
    project=a.wandb_project or 'herp-'+''.join(c.lower() if c.isalnum() else '-' for c in env_id).strip('-')
    run=wandb.init(project=project,name=a.wandb_run_name or f'rq4-{bench}-{env_id}-oracle-s{a.seed}',
                   group=f'rq4-{bench}',config=vars(a),dir=str(out),mode=a.wandb_mode)
a.oracle_samples=a.fragments_per_region
vector=s['args'].get('num_envs',1)>1
if bench=='maniskill':
    device='cuda' if vector else 'cpu';sim='physx_cuda' if vector else 'physx_cpu'
    env_kw=dict(benchmark='maniskill',env_id=env_id,control_mode=a.control_mode,
                reward_mode=a.reward_mode,sim_backend=sim,render_backend='cpu',device=device)
else:
    device='cpu';sim='cpu';vector=False
    env_kw=dict(benchmark=bench,env_id=env_id,device=device)
env=make_adapter(**env_kw).make(a.oracle_samples+a.low_pool if vector else 1,a.seed)
env.reset(seed=a.seed)
agent=Agent(env.obs_dim,env.action_dim).to(device);agent.load_state_dict(s['policy']);agent.eval()
archive=s['archive'];archive._generator.manual_seed(a.seed+1)
collector=(VectorFragmentCollector if vector else FragmentCollector)(env,agent,s['normalizer'],cfg.future_horizon,cfg.action_feature_weight)
generator=torch.Generator().manual_seed(a.seed+7)
ids=[r.region_id for r in archive.regions[1:] if r.snapshots]
rng=np.random.default_rng(a.seed);ids=sorted(rng.choice(ids,min(len(ids),a.regions-1),replace=False).tolist());ids=[0]+ids
X=[];features=[];valid=[];source_ids=[];pred=[];started=time.time()
for rid in ids:
    r=archive.regions[rid];rows=[];masks=[]
    if vector:
        probs=torch.zeros(len(archive),dtype=torch.float64);probs[rid]=1.
        fs=collector.collect_round(archive,probs,s['policy_version'],generator)
        first={}
        for f in fs:first.setdefault(f.slot_id,f)
        for i in range(env.num_envs):
            f=first[i];rows.append(f.traj_features.numpy());masks.append(f.valid_mask.numpy())
    else:
        for k in range(a.oracle_samples+a.low_pool):
            snapshot=None if rid==0 else archive.sample_snapshot(r)
            f=collector.collect(AcquisitionJob(rid,snapshot,cfg.future_horizon),s['policy_version'])
            rows.append(f.traj_features.numpy());masks.append(f.valid_mask.numpy())
    X.append(region_predictor_features(r).numpy());features.append(rows);valid.append(masks);source_ids.append(rid)
    pred.append(s['predictor'].predict(X[-1]))
    progress=dict(region=rid,completed=len(X),total=len(ids),diagnostic_steps=collector.total_steps,seconds=time.time()-started)
    print(json.dumps(progress),flush=True)
    if run:run.log(progress,step=collector.total_steps)
    np.savez_compressed(out/'oracle_pool.tmp.npz',features=np.asarray(features),valid=np.asarray(valid),X=np.asarray(X),
                        region_ids=np.asarray(source_ids),checkpoint_predictions=np.asarray(pred),low_pool=a.low_pool)
    (out/'oracle_pool.tmp.npz').replace(out/'oracle_pool.npz')
meta=dict(checkpoint=str(Path(a.checkpoint).resolve()),checkpoint_train_steps=sum(s['counters'].values()),
          policy_version=s['policy_version'],env_id=s['args']['env_id'],oracle_samples=a.oracle_samples,
          low_pool=a.low_pool,regions=len(ids),seed=a.seed,horizon=cfg.future_horizon,
          diagnostic_steps=collector.total_steps,wall_seconds=time.time()-started,
          sim_backend=sim,frozen_policy=True,frozen_partition=True,disjoint_low_and_oracle_pools=True,
          training_updates=0,censored_fraction=float(1-np.asarray(valid).all(-1).mean()))
(out/'metadata.json').write_text(json.dumps(meta,indent=2))
if run:
    for k,v in meta.items():
        if isinstance(v,(int,float,bool)):run.summary[k]=v
    run.finish()
env.close()
