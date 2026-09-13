"""Reproducible CPU ManiSkill HERP-v3 pilot; identical existing PPO optimizer."""
from __future__ import annotations
import argparse
from collections import defaultdict,deque
from dataclasses import asdict
import json,math,os,random,sys,time,subprocess
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'src')]
import numpy as np
import torch
from train import Agent,Args,ppo_update
from herp.config import HERPV3Config
from herp.archive import RegionArchive
from herp.chain_features import RunningFeatureNormalizer
from herp.v3_observer import PartitionObserver
from herp.vector_acquisition import VectorPartitionObserver,VectorFragmentCollector
from herp.sigma_predictor import LinearVariancePredictor,region_predictor_features,combined_q
from herp.sigma import direct_q_estimate,root_total_variance
from herp.allocator import v3_priority_distribution,allocate_fragments
from herp.acquisition import AcquisitionJob,AcquisitionScheduler,FragmentCollector,concatenate_batches
from herp.gradient_signature import policy_gradient_signature
from herp.envs.maniskill import ManiSkillAdapter
from herp.baselines import RND,Disagreement

METHODS=['ppo','rnd','disagreement','uniform','state_radius_uniform','state_radius_psigma',
         'plr_region','sacl_style','herp_sigma','herp_p','herp']

def json_write(path,data):
    Path(path).write_text(json.dumps(data,indent=2,default=str))

@torch.no_grad()
def evaluate(env,agent,episodes,seed):
    """Vector eval mirroring train.py's proven eval:
    - Track per-slot success_once (max across steps).
    - On done, read info['final_info']["success"] for success_at_end since
      the vec env auto-resets and info["success"] reflects the NEW episode.
    - Report both eval_success (once) and eval_success_final for parity
      with the legacy dispatcher (commit c1c46ac)."""
    obs,_=env.reset(seed=seed)
    n=env.num_envs;dev=env.device
    ep_returns=[];ep_successes=[];ep_successes_final=[]
    ret=torch.zeros(n,device=dev);per_slot_success=torch.zeros(n,device=dev)
    steps=0
    ep_windows=max(1,(episodes+n-1)//n)
    max_steps=max(50,getattr(env,'max_episode_steps',200))*ep_windows*2
    while len(ep_returns)<episodes and steps<max_steps:
        action=agent.act(obs,deterministic=True).clamp(env.action_low(),env.action_high())
        obs,r,term,trunc,info=env.step(action);steps+=1
        ret=ret+r.to(dev).float()
        cur_success=env.success_from_info(info).float().to(dev)
        per_slot_success=torch.maximum(per_slot_success,cur_success)
        done=term.to(dev)|trunc.to(dev)
        if bool(done.any()):
            fi=info.get('final_info') if isinstance(info,dict) else None
            end_success=(env.success_from_info(fi).float().to(dev) if fi is not None else cur_success)
            for i in torch.where(done)[0].tolist():
                ep_returns.append(float(ret[i]))
                ep_successes.append(max(float(per_slot_success[i]),float(end_success[i])))
                ep_successes_final.append(float(end_success[i]))
                ret[i]=0.;per_slot_success[i]=0.
    ep_returns=ep_returns[:episodes];ep_successes=ep_successes[:episodes];ep_successes_final=ep_successes_final[:episodes]
    return dict(eval_return=float(np.mean(ep_returns)) if ep_returns else 0.,
                eval_success=float(np.mean(ep_successes)) if ep_successes else 0.,
                eval_success_final=float(np.mean(ep_successes_final)) if ep_successes_final else 0.,
                episode_returns=ep_returns,episode_successes=ep_successes,
                episode_successes_final=ep_successes_final,
                eval_steps=steps*n,eval_episodes=len(ep_returns))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=METHODS,default='herp')
    p.add_argument('--env-id',default='PickCube-v1'); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--total-timesteps',type=int,default=65536)
    p.add_argument('--batch-size',type=int,default=1024)
    p.add_argument('--num-envs',type=int,default=1)
    p.add_argument('--reference-slots',type=int,default=16)
    p.add_argument('--num-eval-envs',type=int,default=16)
    p.add_argument('--eval-interval',type=int,default=16384);p.add_argument('--eval-episodes',type=int,default=10)
    p.add_argument('--reference-steps',type=int,default=200)
    p.add_argument('--future-horizon',type=int,default=32)
    p.add_argument('--max-regions',type=int,default=64)
    p.add_argument('--chain-radius',type=float,default=2.)
    p.add_argument('--predictor-min-labels',type=int,default=16)
    p.add_argument('--sigma-mode',choices=['shrinkage','direct','predictor'],default='shrinkage')
    p.add_argument('--score-normalize',choices=['none','rank','zscore'],default='rank',
                   help='p and sigma live on incompatible scales; rank-normalize each to [0,1] before multiplying')
    p.add_argument('--score-temperature',type=float,default=1.0)
    p.add_argument('--output-dir',required=True);p.add_argument('--resume-from',default='',
                   help='exact checkpoint path; use --auto-resume to pick latest automatically')
    p.add_argument('--auto-resume',action='store_true',
                   help='if set, load the newest checkpoint_*.pt in --output-dir when --resume-from is empty')
    p.add_argument('--checkpoint-interval',type=int,default=16384)
    p.add_argument('--wandb-mode',choices=['disabled','offline','online'],default='disabled')
    p.add_argument('--wandb-project',default='herp-v3')
    p.add_argument('--wandb-entity',default='')
    p.add_argument('--wandb-group',default='')
    p.add_argument('--wandb-run-name',default='')
    p.add_argument('--wandb-tags',default='')
    p.add_argument('--wandb-log-every',type=int,default=1)
    # Env config knobs — must match the upstream ManiSkill PPO baseline
    # (pd_ee_delta_pose + dense) to reproduce the 100%-success 1M runs;
    # the adapter's own defaults (pd_joint_delta_pos + normalized_dense)
    # are much harder to solve at this budget.
    p.add_argument('--control-mode',default='pd_ee_delta_pose')
    p.add_argument('--reward-mode',default='dense')
    p.add_argument('--obs-mode',default='state')
    args=p.parse_args(argv)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    if not args.resume_from and args.auto_resume:
        candidates=sorted(out.glob('checkpoint_*.pt'),key=lambda x:x.stat().st_mtime)
        if candidates:
            args.resume_from=str(candidates[-1])
            print(f'[auto-resume] loading {args.resume_from}',flush=True)
    torch.set_num_threads(1);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    cfg=HERPV3Config(future_horizon=args.future_horizon,min_common_steps=max(2,args.future_horizon//4),
                    max_regions=args.max_regions,chain_radius=args.chain_radius,predictor_min_labels=args.predictor_min_labels,
                    score_normalize=args.score_normalize,score_temperature=args.score_temperature)
    ppo=Args(num_envs=args.num_envs,num_steps=cfg.future_horizon if args.num_envs>1 else args.batch_size,num_minibatches=8,device='cuda' if args.num_envs>1 else 'cpu',sim_backend='physx_cuda' if args.num_envs>1 else 'physx_cpu')
    vector=args.num_envs>1
    device='cuda' if vector else 'cpu';sim='physx_cuda' if vector else 'physx_cpu'
    if vector and args.total_timesteps%args.num_envs:raise ValueError('Vector budget must be divisible by num_envs')
    env=ManiSkillAdapter(args.env_id,control_mode=args.control_mode,obs_mode=args.obs_mode,reward_mode=args.reward_mode,sim_backend=sim,render_backend='cpu',device=device).make(args.num_envs,args.seed)
    env.reset(seed=args.seed)
    eval_env=ManiSkillAdapter(args.env_id,control_mode=args.control_mode,obs_mode=args.obs_mode,reward_mode=args.reward_mode,sim_backend=sim,render_backend='cpu',device=device,ignore_terminations=True).make(args.num_eval_envs if vector else 1,args.seed+10000)
    agent=Agent(env.obs_dim,env.action_dim).to(device);optimizer=torch.optim.Adam(agent.parameters(),lr=ppo.learning_rate,eps=1e-5)
    normalizer=RunningFeatureNormalizer(eps=1e-3)
    archive=RegionArchive(cfg.max_snapshots_per_region,args.seed);archive.ensure_root()
    observer=(VectorPartitionObserver if vector else PartitionObserver)(env,archive,normalizer,cfg,args.method.startswith('state_radius'))
    collector=(VectorFragmentCollector if vector else FragmentCollector)(env,agent,normalizer,cfg.future_horizon,cfg.action_feature_weight)
    predictor=LinearVariancePredictor(ridge=cfg.predictor_ridge)
    cache=defaultdict(lambda:deque(maxlen=cfg.max_sigma_fragments_per_region))
    child_mean_cache=defaultdict(lambda:deque(maxlen=cfg.max_sigma_fragments_per_region))
    value_observations={}
    scheduler=AcquisitionScheduler(archive,cfg.future_horizon)
    generator=torch.Generator().manual_seed(args.seed+1234)
    intrinsic=RND(env.obs_dim) if args.method=='rnd' else Disagreement(env.obs_dim,env.action_dim) if args.method=='disagreement' else None
    intrinsic_optimizer=torch.optim.Adam(intrinsic.parameters(),lr=3e-4) if intrinsic else None
    ordinary=args.method in ('ppo','rnd','disagreement')
    version=0;activation=None;next_eval=0;next_checkpoint=args.checkpoint_interval
    started=time.time();eval_steps=0
    wandb_run_id_from_ckpt=None
    if args.resume_from:
        state=torch.load(args.resume_from,weights_only=False,map_location='cpu')
        if state['args']['method']!=args.method or state['args']['env_id']!=args.env_id:
            raise ValueError('Resume method/task mismatch')
        agent.load_state_dict(state['policy']);optimizer.load_state_dict(state['optimizer'])
        normalizer=state['normalizer'];archive=state['archive'];predictor=state['predictor']
        observer=state['observer'];observer.adapter=env;observer.archive=archive;observer.regionizer.archive=archive
        collector.normalizer=normalizer;collector.counters=state['counters'];collector.fragment_id=state['fragment_id']
        collector.observer=observer;scheduler.archive=archive
        for rid,fs in state['sigma_cache'].items():cache[rid].extend(fs)
        for rid,fs in state.get('child_mean_cache',{}).items():child_mean_cache[rid].extend(fs)
        value_observations=state.get('value_observations',{})
        version=state['policy_version'];activation=state['activation'];next_eval=collector.total_steps+args.eval_interval
        next_checkpoint=collector.total_steps+args.checkpoint_interval
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng']);generator.set_state(state['allocator_rng'])
        if intrinsic:
            intrinsic.load_state_dict(state['intrinsic']);intrinsic_optimizer.load_state_dict(state['intrinsic_optimizer'])
        wandb_run_id_from_ckpt=state.get('wandb_run_id')
    if (out/'summary.json').exists() and not args.resume_from:
        print(json.dumps({'skip':True,'reason':'summary.json already present','output_dir':str(out)}),flush=True)
        return
    wandb_run=None;wandb_module=None
    if args.wandb_mode!='disabled':
        try:
            import wandb as wandb_module
        except ImportError as exc:
            raise RuntimeError('W&B logging requested but wandb is not installed') from exc
        run_name=args.wandb_run_name or f'{args.method}-{Path(args.output_dir).name}-s{args.seed}'
        tags=[t.strip() for t in args.wandb_tags.split(',') if t.strip()]
        wandb_run=wandb_module.init(project=args.wandb_project,entity=args.wandb_entity or None,
            group=args.wandb_group or None,name=run_name,tags=tags or None,
            config={**vars(args),'herp':asdict(cfg),'ppo':asdict(ppo)},
            dir=str(out),mode=args.wandb_mode,id=wandb_run_id_from_ckpt,
            resume='allow' if wandb_run_id_from_ckpt else None)
        json_write(out/'wandb.json',dict(id=wandb_run.id,name=wandb_run.name,project=args.wandb_project,
            entity=args.wandb_entity or None,group=args.wandb_group or None,mode=args.wandb_mode,
            url=getattr(wandb_run,'url',None)))
        wandb_run.define_metric('global_env_steps')
        wandb_run.define_metric('*',step_metric='global_env_steps')
    json_write(out/'config.json',dict(args=vars(args),herp=asdict(cfg),ppo=asdict(ppo)))
    import importlib.metadata as md
    import hashlib, tarfile
    source_root=Path(__file__).resolve().parents[1]
    source_files=sorted((source_root/'src/herp').rglob('*.py'))+[Path(__file__).resolve(),source_root/'train.py']
    json_write(out/'source_hashes.json',{str(f.relative_to(source_root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files})
    with tarfile.open(out/'source.tar.gz','w:gz') as tar:
        for f in source_files:tar.add(f,arcname=str(f.relative_to(source_root)))
    json_write(out/'provenance.json',dict(python=sys.executable,git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        packages={x:md.version(x) for x in ['torch','mani-skill','sapien','gymnasium','numpy']},
        normalization='running; centroids and cached futures rebased together each round',sim_backend=sim,
        source_snapshot='source.tar.gz',root_total_variance='child-entry proxy; direct root logged separately',
        sacl_style='value change on fixed archived representative state; uncertainty coefficient zero'))
    metrics_file=open(out/'metrics.jsonl','a',buffering=1);regions_file=open(out/'regions.jsonl','a',buffering=1)

    def save_checkpoint(name):
        adapter=observer.adapter;observer.adapter=None
        try:
            state=dict(policy=agent.state_dict(),optimizer=optimizer.state_dict(),normalizer=normalizer,
                archive=archive,observer=observer,predictor=predictor,sigma_cache=dict(cache),
                child_mean_cache=dict(child_mean_cache),value_observations=value_observations,
                counters=collector.counters,fragment_id=collector.fragment_id,policy_version=version,
                activation=activation,args=vars(args),cfg=cfg,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(),allocator_rng=generator.get_state(),
                intrinsic=None if intrinsic is None else intrinsic.state_dict(),
                intrinsic_optimizer=None if intrinsic is None else intrinsic_optimizer.state_dict(),
                wandb_run_id=None if wandb_run is None else wandb_run.id)
            torch.save(state,out/(name+".tmp"))
            (out/(name+".tmp")).replace(out/name)
        finally: observer.adapter=adapter

    def wandb_log(kind, payload):
        if wandb_run is None: return
        clean={f'{kind}/{k}':float(v) for k,v in payload.items()
               if isinstance(v,(int,float)) and math.isfinite(v)}
        clean['global_env_steps']=int(collector.total_steps)
        wandb_run.log(clean)

    while collector.total_steps<args.total_timesteps:
        if collector.total_steps>=next_eval:
            ev=evaluate(eval_env,agent,args.eval_episodes,args.seed+10000)
            eval_steps+=ev['eval_steps'];ev.update(step=collector.total_steps,policy_version=version,wall_seconds=time.time()-started,type='evaluation')
            metrics_file.write(json.dumps(ev)+'\n');print(json.dumps({k:v for k,v in ev.items() if not k.startswith('episode_')}),flush=True)
            wandb_log('eval',{k:v for k,v in ev.items() if not k.startswith('episode_') and k not in ('type',)})
            next_eval=collector.total_steps+args.eval_interval
        remaining=min(args.num_envs*cfg.future_horizon if vector else args.batch_size,args.total_timesteps-collector.total_steps)
        batch=[];fresh=defaultdict(list);pre_x={r.region_id:region_predictor_features(r) for r in archive}
        warmup=(len(archive)-1<cfg.min_non_root_regions or len(predictor.y)<cfg.predictor_min_labels)
        if not warmup and activation is None:activation=collector.total_steps
        collection_start=time.time()
        if vector:
            regions=archive.regions
            probs=v3_priority_distribution(regions,cfg,args.method)
            nref=0 if ordinary or version==0 else min(args.reference_slots,args.num_envs-1)
            batch=collector.collect_round(archive,probs,version,generator,nref,ordinary or version==0,warmup,remaining//args.num_envs)
            for_ref=[f for f in batch if f.category=='REFERENCE']
            allocations={r.region_id:0 for r in regions}
            for f in batch:
                allocations[f.source_region_id]=allocations.get(f.source_region_id,0)+1
                if (f.root_started or f.source_region_id>0) and len(f.rewards)==cfg.future_horizon:
                    fresh[f.source_region_id].append(f)
        else:
            # Ordinary-reset reference traverses whole episodes, allowing discovery at
            # later task stages. Reference is counted and also used in the PPO update.
            nref=0 if ordinary or version==0 else min(args.reference_steps,remaining)
            for_ref=[]
            while nref>0:
                reset=not for_ref or bool(for_ref[-1].terminated[-1]|for_ref[-1].truncated[-1])
                f=collector.collect(AcquisitionJob(0,max_steps=min(cfg.future_horizon,nref)),version,'REFERENCE',reset=reset)
                for_ref.append(f);batch.append(f);nref-=len(f.rewards);remaining-=len(f.rewards)
                if f.root_started and len(f.rewards)==cfg.future_horizon:fresh[0].append(f)
            regions=archive.regions
            probs=v3_priority_distribution(regions,cfg,args.method)
            allocations={r.region_id:0 for r in regions}
            while remaining>0:
                if ordinary or version==0:
                    # Ordinary PPO continues its episodes across rollout batches.
                    reset=collector.obs is None or (batch and bool(batch[-1].terminated[-1]|batch[-1].truncated[-1]))
                    rid=0;snap=None
                else:
                    rid=0 if warmup else int(torch.multinomial(probs,1,generator=generator))
                    snap=None if rid==0 else archive.sample_snapshot(archive.regions[rid]);reset=True
                f=collector.collect(AcquisitionJob(rid,snap,min(cfg.future_horizon,remaining)),version,reset=reset)
                batch.append(f);remaining-=len(f.rewards)
                allocations[rid]=allocations.get(rid,0)+1
                if (f.root_started or rid>0) and len(f.rewards)==cfg.future_horizon:fresh[rid].append(f)
        collection_seconds=time.time()-collection_start
        processing_start=time.time()
        if version==0:
            normalizer.update(torch.cat([f.states for f in batch]))
            collector.observer=None if ordinary else observer
            # Normalize already collected first-batch futures with the newly
            # fitted, subsequently frozen transform before storing any labels.
            for f in batch:
                n=len(f.rewards);f.traj_features[:n,:env.obs_dim]=normalizer.normalize(f.next_states)
        else:
            old_mean=normalizer.mean.clone()
            old_scale=(normalizer.m2/normalizer.count).sqrt().clamp_min(normalizer.eps)
            normalizer.update(torch.cat([f.states for f in batch]))
            def rebase(z):
                return normalizer.normalize(z*old_scale.to(z)+old_mean.to(z))
            for r in archive.regions[1:]:
                r.centroid=rebase(r.centroid.reshape(2,-1)).flatten()
            if vector:observer.rebase_previous(rebase)
            elif observer.previous is not None:
                z,mu,ls=observer.previous;observer.previous=(rebase(z),mu,ls)
            for rows in child_mean_cache.values():
                for _,features in rows:
                    features[:,:env.obs_dim]=rebase(features[:,:env.obs_dim])
            for f in batch+[f for fs in cache.values() for f in fs]:
                n=len(f.rewards)
                f.state_features=normalizer.normalize(f.next_states)
                f.traj_features[:n,:env.obs_dim]=f.state_features
        if intrinsic:
            for f in batch:
                z=normalizer.normalize(f.states);zn=normalizer.normalize(f.next_states)
                with torch.no_grad():bonus=intrinsic.bonus(z,f.actions,zn)
                f.rewards=f.rewards+ppo.intrinsic_coef*bonus
            ib=concatenate_raw(batch)
            for _ in range(ppo.intrinsic_epochs):
                loss=intrinsic.loss(normalizer.normalize(ib['states']),ib['actions'],normalizer.normalize(ib['next_states']))
                intrinsic_optimizer.zero_grad();loss.backward();intrinsic_optimizer.step()
        for f in batch:f.gae(ppo.gamma,ppo.gae_lambda)
        merged={k:v.to(device) for k,v in concatenate_batches(batch).items()}
        # Signed cosine EMA as THEORY §20, with one common advantage scale.
        if for_ref:
            ref={k:v.to(device) for k,v in concatenate_batches(for_ref).items()};scale=float(merged['advantages'].std().clamp_min(1e-8))
            gref=policy_gradient_signature(agent,ref['obs'],ref['actions'],ref['advantages'],scale)
            groups=defaultdict(list)
            for f in batch:
                if f.category!='REFERENCE':groups[f.source_region_id].append(f)
            for rid,fs in groups.items():
                b={k:v.to(device) for k,v in concatenate_batches(fs).items()};g=policy_gradient_signature(agent,b['obs'],b['actions'],b['advantages'],scale)
                raw=float(torch.dot(g,gref)/(g.norm()*gref.norm()+1e-8));r=archive.regions[rid]
                r.p_raw=raw;r.p_ema=cfg.relevance_ema_tau*r.p_ema+(1-cfg.relevance_ema_tau)*raw
                r.last_scored_step=collector.total_steps
        for rid,fs in fresh.items():
            cache[rid].extend(fs)
            q,pairs=direct_q_estimate(fs,cfg.min_common_steps)
            if rid in pre_x and math.isfinite(q):predictor.add_label(pre_x[rid],q,len(fs),version)
        if len(predictor.y)>=cfg.predictor_min_labels:predictor.fit()
        # Root children start at ordinary resets. Their observed fixed-M
        # conditional means are available even before separate region jobs.
        for rid,fs in fresh.items():
            for f in fs:
                child=rid if rid>0 else int(f.chain_region_id[0])
                if child>0:
                    child_mean_cache[child].append((version,f.traj_features.clone()))
        current_q={};means={rid:torch.stack([x for _,x in rows]).mean(0).flatten()/math.sqrt(cfg.future_horizon)
                             for rid,rows in child_mean_cache.items() if rows}
        for r in archive:
            recent=[f for f in cache[r.region_id] if version-f.policy_version<=cfg.max_sigma_policy_lag]
            q,pairs=direct_q_estimate(recent,cfg.min_common_steps)
            r.q_direct=q;r.sigma_sample_count=len(recent);current_q[r.region_id]=q
            if recent:means[r.region_id]=torch.stack([f.traj_features for f in recent]).mean(0).flatten()/math.sqrt(cfg.future_horizon)
        finite=[q for q in current_q.values() if math.isfinite(q)];prior=float(np.median(finite)) if finite else 0.
        for r in archive:
            r.q_pred=predictor.predict(region_predictor_features(r)) if predictor.coef is not None else prior
            r.q_combined=combined_q(r.q_direct,r.q_pred,r.sigma_sample_count,cfg.sigma_predictor_kappa)
            if args.sigma_mode=='direct':r.q_combined=max(0.,r.q_direct) if math.isfinite(r.q_direct) else prior
            elif args.sigma_mode=='predictor':r.q_combined=r.q_pred
            r.sigma_raw=math.sqrt(r.q_combined+cfg.sigma_floor**2)
        weights=observer.graph.child_probabilities(0)
        root_q=root_total_variance(weights,means,{r.region_id:r.q_combined for r in archive})
        # Child means are unavailable before their first acquisition. A common
        # prior is explicitly flagged; never silently renormalize away children.
        missing_root_means=bool(weights and not math.isfinite(root_q))
        if missing_root_means:root_q=prior
        archive.regions[0].q_combined=root_q;archive.regions[0].sigma_raw=math.sqrt(root_q+cfg.sigma_floor**2)
        for r in archive:
            fs=[f for f in batch if f.source_region_id==r.region_id]
            if fs:
                td=torch.cat([f.rewards+ppo.gamma*f.next_values*(~f.terminated)-f.values for f in fs]).abs().mean()
                r.td_error_ema=.9*r.td_error_ema+.1*float(td)
            if r.region_id not in value_observations:
                if r.snapshots:value_observations[r.region_id]=r.snapshots[0].obs.clone()
                elif r.is_root:value_observations[0]=batch[0].states[0].clone()
            if r.region_id in value_observations:
                with torch.no_grad():value=float(agent.get_value(value_observations[r.region_id][None].to(device)))
                r.value_change=0. if r.previous_value is None else abs(value-r.previous_value)
                r.previous_value=value
        processing_seconds=time.time()-processing_start
        update_start=time.time()
        losses=ppo_update(agent,optimizer,merged,ppo);version+=1
        update_seconds=time.time()-update_start
        assert collector.total_steps<=args.total_timesteps
        record=dict(type='training',step=collector.total_steps,policy_version=version,method=args.method,
            wall_seconds=time.time()-started,collection_seconds=collection_seconds,processing_seconds=processing_seconds,update_seconds=update_seconds,budget=collector.counters.copy(),num_regions=len(archive),
            num_chains=observer.chains,predictor_labels=len(predictor.y),activation_step=activation,
            root_direct=archive.regions[0].q_direct,root_total_variance=root_q,root_missing_child_means=missing_root_means,
            root_fraction=allocations.get(0,0)/max(1,sum(allocations.values())),
            allocation_entropy=float(-torch.special.xlogy(probs,probs.clamp_min(1e-12)).sum()),
            losses=losses)
        metrics_file.write(json.dumps(record)+'\n')
        for r in archive:
            row={k:v for k,v in vars(r).items() if k not in ('centroid','snapshots')}
            row.update(step=collector.total_steps,policy_version=version,allocated_fragments=allocations.get(r.region_id,0))
            regions_file.write(json.dumps(row)+'\n')
        if version%8==0:print(json.dumps(record),flush=True)
        if wandb_run is not None and version%args.wandb_log_every==0:
            flat={k:v for k,v in record.items() if isinstance(v,(int,float))}
            for k,v in (record.get('losses') or {}).items():
                if isinstance(v,(int,float)):flat[f'loss_{k}']=v
            for k,v in (record.get('budget') or {}).items():
                if isinstance(v,(int,float)):flat[f'budget_{k.lower()}']=v
            sigmas=[float(r.sigma_raw) for r in archive if r.sigma_raw>0]
            pemas=[float(max(0.,r.p_ema)) for r in archive]
            if sigmas:flat.update(sigma_mean=float(np.mean(sigmas)),sigma_std=float(np.std(sigmas)),sigma_max=float(np.max(sigmas)))
            if pemas:flat.update(p_mean=float(np.mean(pemas)),p_std=float(np.std(pemas)),p_max=float(np.max(pemas)))
            wandb_log('train',flat)
        if collector.total_steps>=next_checkpoint:
            save_checkpoint(f'checkpoint_{collector.total_steps}.pt');next_checkpoint+=args.checkpoint_interval
    ev=evaluate(eval_env,agent,args.eval_episodes,args.seed+10000);eval_steps+=ev['eval_steps']
    ev.update(step=collector.total_steps,policy_version=version,type='evaluation',wall_seconds=time.time()-started)
    metrics_file.write(json.dumps(ev)+'\n');print(json.dumps(ev),flush=True)
    wandb_log('eval',{k:v for k,v in ev.items() if not k.startswith('episode_') and k not in ('type',)})
    save_checkpoint('checkpoint_final.pt')
    json_write(out/'partition_events.json',observer.events)
    json_write(out/'chain_lengths.json',observer.completed_lengths)
    json_write(out/'summary.json',dict(status='completed',method=args.method,env_id=args.env_id,seed=args.seed,
        training_steps=collector.total_steps,budget=collector.counters,eval_steps=eval_steps,
        final_evaluation=ev,activation_step=activation,wall_seconds=time.time()-started))
    if wandb_run is not None:
        wandb_run.summary['final/success']=ev['eval_success']
        wandb_run.summary['final/return']=ev['eval_return']
        wandb_run.summary['final/global_env_steps']=collector.total_steps
        wandb_run.summary['final/wall_seconds']=time.time()-started
        wandb_run.finish()
    env.close();eval_env.close();metrics_file.close();regions_file.close()

def concatenate_raw(fs):
    return {k:torch.cat([getattr(f,k) for f in fs]) for k in ('states','actions','next_states')}

if __name__=='__main__':main()
