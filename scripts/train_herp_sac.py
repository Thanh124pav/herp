"""Serial HERP-SAC and its matched uniform-replay SAC control.

Native SAC updates are shared and tested against ManiSkill upstream. Acquisition
alone differs. Calibration/reference interactions enter replay and count in budget.
"""
import os
os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/lvp_icd.json")
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
import tyro
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'src')]
from scripts.sac_official import Args as NativeArgs
from scripts.train_v3 import evaluate
from herp.config import HERPV3Config
from herp.envs.maniskill import ManiSkillAdapter
from herp.learners.sac import SACLearner
from herp.allocation_controller import AllocationController
from herp.provenance import save_provenance


@dataclass
class Args(NativeArgs):
    seed: int = 0
    method: str = 'sac_herp'
    num_envs: int = 1
    num_eval_envs: int = 1
    buffer_device: str = 'cpu'
    cuda: bool = False
    control_mode: str = 'pd_ee_delta_pose'
    output_dir: str = 'outputs/sac_herp'
    future_horizon: int = 32
    sigma_mode: str = 'shrinkage'
    sigma_kappa: float = 8.
    eval_episodes: int = 10
    eval_freq: int = 10000
    wandb_mode: str = 'online'
    wandb_project_name: str = 'herp-framework'
    phase: str = 'pilot'
    resume_from: str = ''


def main():
    args=tyro.cli(Args)
    if args.method not in ('sac','sac_herp','sac_herp_p','sac_herp_sigma'):
        raise ValueError('Unsupported method')
    if args.num_envs!=1 or args.num_eval_envs!=1:raise ValueError('This acquisition runner requires serial single environments')
    if args.future_horizon<2 or args.training_freq<1 or args.total_timesteps<1 or args.eval_freq<1 or args.eval_episodes<1:
        raise ValueError('Invalid horizon, budget or evaluation protocol')
    if args.sigma_mode not in ('direct','predictor','shrinkage') or args.wandb_mode not in ('online','offline','disabled'):
        raise ValueError('Invalid estimator/logging mode')
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.jsonl').exists() and not args.resume_from:raise FileExistsError('Use a new output directory or explicit checkpoint resume')
    save_provenance(out,Path(__file__).resolve().parents[1])
    torch.set_num_threads(1);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device='cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    env=ManiSkillAdapter(args.env_id,control_mode=args.control_mode,sim_backend='physx_cpu',render_backend='cpu',device=device,ignore_terminations=not args.partial_reset).make(1,args.seed)
    ev_env=ManiSkillAdapter(args.env_id,control_mode=args.control_mode,sim_backend='physx_cpu',render_backend='cpu',device=device,ignore_terminations=True).make(1,args.seed+10000)
    learner=SACLearner(env.env,args,device)
    cfg=HERPV3Config(future_horizon=args.future_horizon,min_common_steps=max(2,args.future_horizon//4),sigma_predictor_kappa=args.sigma_kappa)
    controller=AllocationController(env,cfg,args.seed,args.sigma_mode)
    counts=dict(ROOT_ACQUISITION=0,REGION_ACQUISITION=0,REFERENCE=0)
    steps=version=eval_steps=0;next_eval=0;obs=None;learning=False;done=True
    run=None;started=time.time()
    if args.resume_from:
        checkpoint=Path(args.resume_from)
        state=torch.load(checkpoint,weights_only=False,map_location='cpu')
        for key in ('method','env_id','seed','future_horizon','sigma_mode','sigma_kappa','training_freq','utd'):
            if state['args'][key]!=getattr(args,key):raise ValueError(f'Resume mismatch: {key}')
        learner.load(checkpoint.with_suffix('.learner.pt'))
        controller=state['controller'];controller.adapter=env;controller.observer.adapter=env
        counts=state.get('counts',dict(PRE_RESUME_UNCLASSIFIED=state['steps'],ROOT_ACQUISITION=0,REGION_ACQUISITION=0,REFERENCE=0))
        steps=state['steps'];version=state['version'];eval_steps=state['eval_steps'];learning=state['learning']
        obs=env.restore_state(torch.tensor([0]),state['env_state']);done=state['done']
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);next_eval=steps+args.eval_freq
    else:
        state={}
    (out/'config.json').write_text(json.dumps({**asdict(args),'herp':asdict(cfg),'normalization':'counted warmup calibration then frozen','root_sigma':'direct ordinary-reset futures','reward_mode':'normalized_dense'},indent=2))
    if args.wandb_mode!='disabled':
        import wandb
        run=wandb.init(project=args.wandb_project_name,entity=args.wandb_entity,name=f'{args.method}-{args.env_id}-s{args.seed}',
            config=asdict(args),dir=str(out),mode=args.wandb_mode,id=state.get('wandb_id'),resume='allow' if state.get('wandb_id') else None)
        run.define_metric('env_steps');run.define_metric('*',step_metric='env_steps')
    metrics=(out/'metrics.jsonl').open('a',buffering=1)
    regions=(out/'regions.jsonl').open('a',buffering=1)

    def save_checkpoint():
        path=out/f'checkpoint_{steps}.pt'
        learner.save(path.with_suffix('.learner.pt'))
        controller.adapter=None;controller.observer.adapter=None
        try:
            torch.save(dict(args=asdict(args),controller=controller,steps=steps,version=version,eval_steps=eval_steps,
                learning=learning,done=done,counts=counts,env_state=env.save_state(torch.tensor([0])),
                python_rng=random.getstate(),numpy_rng=np.random.get_state(),wandb_id=run.id if run else None),path.with_suffix('.tmp'))
            path.with_suffix('.tmp').replace(path)
        finally:controller.adapter=env;controller.observer.adapter=env

    def eval_and_log():
        nonlocal eval_steps
        # Evaluation cannot perturb training policy/replay randomness.
        with torch.random.fork_rng(devices=[0] if device=='cuda' else []):
            row=evaluate(ev_env,learner,args.eval_episodes,args.seed+10000)
        if row['eval_episodes']!=args.eval_episodes:raise RuntimeError('Incomplete evaluation')
        eval_steps+=row['eval_steps']
        row.update(type='evaluation',step=steps)
        metrics.write(json.dumps(row)+'\n')
        print(json.dumps({k:v for k,v in row.items() if not k.startswith('episode_')}),flush=True)
        if run:run.log({'env_steps':steps,**{f'eval/{k}':v for k,v in row.items() if isinstance(v,(float,int))}})

    try:
        while steps<args.total_timesteps:
            if steps>=next_eval:
                eval_and_log();next_eval=steps+args.eval_freq
            start_steps=steps;fragments=[];reference=[];pre=controller.begin_round()
            remaining=min(args.training_freq,args.total_timesteps-steps)
            while remaining:
                use_herp=args.method!='sac' and learning
                reference_job=use_herp and not fragments
                if use_herp:
                    if reference_job:rid,snapshot=0,None
                    else:rid,snapshot,_=controller.choose(args.method.replace('sac_',''))
                    obs=env.reset()[0] if rid==0 else env.restore_state(torch.tensor([0]),[snapshot.env_state])
                    if snapshot is not None and not torch.allclose(obs[0].cpu(),snapshot.obs,atol=1e-4,rtol=0):
                        raise RuntimeError('Restored observation mismatch')
                    root_started=rid==0;controller.observer.new_fragment(rid,root_started)
                else:
                    rid=0;root_started=obs is None or done
                    if root_started:obs=env.reset()[0]
                states=[];next_states=[];actions_list=[]
                for _ in range(min(args.future_horizon,remaining)):
                    with torch.no_grad():
                        action=learner.act(obs) if learning else 2*torch.rand((1,env.action_dim),device=device)-1
                        if use_herp:
                            mean,logstd=learner.actor(obs)
                            controller.observer.observe(obs[0],mean[0],logstd[0],steps)
                    nxt,reward,term,trunc,info=env.step(action)
                    done=bool(term[0]|trunc[0]);actual=info['final_observation'] if done and 'final_observation' in info else nxt
                    stop=(term|trunc) if args.bootstrap_at_done=='never' else torch.zeros_like(term) if args.bootstrap_at_done=='always' else term
                    learner.observe(dict(obs=obs,next_obs=actual,action=action,reward=reward,done=stop,region=rid))
                    states.append(obs.detach().cpu());next_states.append(actual.detach().cpu());actions_list.append(action.detach().cpu())
                    if not controller.active:controller.calibrate(obs.cpu())
                    category='REFERENCE' if reference_job else 'REGION_ACQUISITION' if rid else 'ROOT_ACQUISITION'
                    counts[category]+=1
                    obs=nxt;steps+=1;remaining-=1
                    if done:
                        if use_herp:controller.observer.end_episode()
                        break
                if reference_job:reference.extend(states)
                fragments.append(controller.fragment(rid,torch.cat(states),torch.cat(next_states),torch.cat(actions_list),version,root_started))
            diagnostics={}
            if args.method!='sac' and learning:
                diagnostics=controller.finish_round(fragments,torch.cat(reference) if reference else torch.empty(0,env.obs_dim),learner,version,pre)
            losses={}
            if steps>=args.learning_starts:
                learning=True;controller.active=True
                for _ in range(int((steps-start_steps)*args.utd)):losses=learner.update()
            version+=1
            assert sum(counts.values())==steps
            row=dict(type='training',step=steps,budget=counts.copy(),losses=losses,**diagnostics)
            metrics.write(json.dumps(row)+'\n')
            if run:run.log({'env_steps':steps,**{f'loss/{k}':v for k,v in losses.items()},**{f'herp/{k}':v for k,v in diagnostics.items()},**{f'budget/{k}':v for k,v in counts.items()}})
            if version%16==0:
                occupancy=learner.replay.occupancy()
                for r in controller.archive:
                    row=dict(step=steps,region=r.region_id,p_raw=r.p_raw,p_ema=r.p_ema,q_direct=r.q_direct,q_pred=r.q_pred,
                        sigma=r.sigma_raw,actor_gradient_norm=getattr(r,'actor_gradient_norm',0.),reference_gradient_norm=getattr(r,'reference_gradient_norm',0.),new_transitions=learner.replay.acquired[r.region_id],buffer_occupancy=occupancy.get(r.region_id,0),replay_samples=learner.replay.sampled[r.region_id])
                    regions.write(json.dumps(row)+'\n')
                    if run:run.log({'env_steps':steps,**{f'region/{r.region_id}/{k}':v for k,v in row.items() if isinstance(v,(int,float)) and math.isfinite(v)}})
            if steps>=next_eval:save_checkpoint()
        eval_and_log();save_checkpoint()
        (out/'summary.json').write_text(json.dumps(dict(status='completed',method=args.method,env_steps=steps,eval_env_steps=eval_steps,wall_seconds=time.time()-started)))
    finally:
        metrics.close();regions.close();env.close();ev_env.close()
        if run:run.finish()

if __name__=='__main__':main()
