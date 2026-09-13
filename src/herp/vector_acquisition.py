"""Synchronous vector acquisition with exact accounting of every stepped slot.

Independent fragments end at termination, time limits, or the PPO round edge.
A slot that ends early receives another allocated job, never uncounted padding.
"""
import math
import numpy as np
import torch
from .archive import Snapshot
from .chain_features import RunningFeatureNormalizer,symmetric_gaussian_kl
from .chain_partition import ChainRegionizer
from .region_graph import RegionGraph
from .acquisition import RolloutFragment

class VectorPartitionObserver:
    def __init__(self,adapter,archive,normalizer,cfg,state_radius=False):
        self.adapter,self.archive,self.normalizer,self.cfg=adapter,archive,normalizer,cfg
        self.regionizer=ChainRegionizer(archive,cfg);self.graph=RegionGraph();self.state_radius=state_radius
        n=adapter.num_envs
        self.previous_z=None;self.previous_mu=None;self.previous_ls=None
        self.has_previous=torch.zeros(n,dtype=torch.bool)
        self.current=torch.full((n,),-1,dtype=torch.long);self.lengths=torch.zeros(n,dtype=torch.long)
        self.sources=torch.zeros(n,dtype=torch.long)
        self.channels=RunningFeatureNormalizer();self.history=torch.empty(cfg.boundary_score_buffer,2,dtype=torch.float64)
        self.history_n=0;self.cursor=0;self.threshold=float('inf')
        self.chains=0;self.events=[];self.completed_lengths=[]

    def close_slots(self,ids,censored):
        for i in ids.tolist():
            rid=int(self.current[i]);length=int(self.lengths[i])
            if rid>=0 and length:
                r=self.archive.regions[rid]
                r.mean_chain_len+=(length-r.mean_chain_len)/max(1,r.num_chains)
                self.completed_lengths.append((length,censored))
        self.lengths[ids]=0

    def new_jobs(self,ids,sources):
        ids=ids.cpu();self.close_slots(ids,True)
        self.current[ids]=-1;self.has_previous[ids]=False;self.sources[ids]=sources.cpu()

    def rebase_previous(self,rebase):
        if self.previous_z is not None:self.previous_z=rebase(self.previous_z)

    def observe(self,obs,mu,ls,step):
        obs,mu,ls=obs.detach().cpu(),mu.detach().cpu(),ls.detach().cpu()
        z=self.normalizer.normalize(obs)
        n=len(z);pc=torch.zeros(n);sc=torch.zeros(n);score=torch.zeros(n,dtype=torch.float64)
        if self.previous_z is not None:
            pc=symmetric_gaussian_kl(self.previous_mu,self.previous_ls,mu,ls)
            sc=(z-self.previous_z).norm(dim=-1)
        pc[~self.has_previous]=0.;sc[~self.has_previous]=0.
        raw=torch.stack([pc,sc],-1).double();valid=raw[self.has_previous]
        self.channels.update(valid)
        count=min(self.history_n,len(self.history))
        weights=torch.tensor([self.cfg.boundary_lambda_policy,self.cfg.boundary_lambda_state],dtype=torch.float64)
        if count>=16:
            self.threshold=float(torch.quantile(self.channels.normalize(self.history[:count])@weights,self.cfg.boundary_percentile))
            score=self.channels.normalize(raw)@weights
        boundary=(score>self.threshold)&(self.lengths>=self.cfg.boundary_min_chain_len)&self.has_previous
        if self.state_radius:boundary[:]=True
        entries=torch.where(boundary|(self.current<0))[0]
        if len(entries):
            snapshots=self.adapter.save_state(entries)
            elapsed=self.adapter.elapsed_steps().cpu()
            self.close_slots(entries,False)
            for j,i in enumerate(entries.tolist()):
                prev=z[i] if not self.has_previous[i] or self.state_radius else self.previous_z[i]
                snap=Snapshot(snapshots[j],obs[i].clone(),step+i,elapsed_steps=int(elapsed[i]))
                entropy=float((ls[i]+.5*math.log(2*math.pi*math.e)).sum())
                old=int(self.current[i])
                rid=self.regionizer.assign_entry(torch.cat([prev,z[i]]),snap,step+i,float(pc[i]),entropy,float(sc[i]))
                self.graph.observe_path([int(self.sources[i]) if old<0 else old,rid]);self.current[i]=rid;self.chains+=1
                self.events.append(dict(step=step+i,region=rid,policy_change=float(pc[i]),state_change=float(sc[i]),
                    score=float(score[i]),threshold=self.threshold,elapsed=int(elapsed[i]),fragment_start=old<0))
        # Batched causal threshold: each vector step uses only earlier steps.
        for row in valid:
            self.history[self.cursor]=row;self.cursor=(self.cursor+1)%len(self.history);self.history_n+=1
        self.lengths+=1;self.has_previous[:]=True
        self.previous_z,self.previous_mu,self.previous_ls=z,mu,ls
        return self.current.clone()

class VectorFragmentCollector:
    def __init__(self,adapter,agent,normalizer,horizon=32,action_weight=1.,observer=None):
        self.adapter,self.agent,self.normalizer=adapter,agent,normalizer
        self.horizon,self.action_weight,self.observer=horizon,action_weight,observer
        self.obs=None;self.fragment_id=0
        self.counters=dict(ROOT_ACQUISITION=0,REGION_ACQUISITION=0,REFERENCE=0)
        self.root_started=None

    @property
    def total_steps(self):return sum(self.counters.values())

    def collect_round(self,archive,probs,version,generator,reference_slots=0,ordinary=False,warmup=False,steps=None):
        n=self.adapter.num_envs;steps=steps or self.horizon
        ids=torch.arange(n);device=self.adapter.device
        is_ref=torch.arange(n)<reference_slots
        sources=torch.zeros(n,dtype=torch.long) if ordinary or warmup else torch.multinomial(probs,n,replacement=True,generator=generator)
        sources[is_ref]=0
        started=torch.zeros(n,dtype=torch.bool) if self.root_started is None else self.root_started.clone()
        # Entire round is stepped, so its budget is known exactly in advance.
        def start_jobs(which):
            nonlocal started
            roots=which[sources[which]==0];local=which[sources[which]>0]
            if self.obs is None:self.obs,_=self.adapter.reset()
            if len(roots):
                current,_=self.adapter.reset_indices(roots);self.obs[roots.to(device)]=current[roots.to(device)]
            if len(local):
                snaps=[archive.sample_snapshot(archive.regions[int(sources[i])]) for i in local]
                restored=self.adapter.restore_state(local,[s.env_state for s in snaps])
                error=float((restored.cpu()-torch.stack([s.obs for s in snaps])).abs().max())
                if error>1e-4:raise RuntimeError(f'GPU snapshot mismatch {error}')
                self.obs[local.to(device)]=restored
            started[which]=True
            if self.observer:self.observer.new_jobs(which,sources[which])
        if self.obs is None or not ordinary:start_jobs(ids)
        # Preallocate time-major buffers; CPU transfer only once per vector step.
        buffers={k:[] for k in ('obs','actions','rewards','logprobs','values','term','trunc','next','next_values','regions','applied','sources','starts')}
        low,high=self.adapter.action_low(),self.adapter.action_high()
        for t in range(steps):
            with torch.no_grad():
                dist=self.agent.get_distribution(self.obs)
                action,lp,_,value=self.agent.get_action_and_value(self.obs)
            regions=self.observer.observe(self.obs,dist.mean,self.agent.logstd.expand_as(dist.mean),self.total_steps) if self.observer else torch.zeros(n,dtype=torch.long)
            applied=action.clamp(low,high)
            nxt,reward,term,trunc,info=self.adapter.step(applied)
            actual_next=nxt.clone();done=term|trunc
            if bool(done.any()) and 'final_observation' in info:actual_next[done]=info['final_observation'][done]
            with torch.no_grad():nv=self.agent.get_value(actual_next)
            row=(self.obs,action,reward,lp,value,term,trunc,actual_next,nv,regions,applied,sources,started)
            for k,x in zip(buffers,row):buffers[k].append(x.detach().cpu().clone())
            self.counters['REFERENCE']+=int(is_ref.sum())
            self.counters['ROOT_ACQUISITION']+=int(((sources==0)&~is_ref).sum())
            self.counters['REGION_ACQUISITION']+=int(((sources>0)&~is_ref).sum())
            self.obs=nxt;started[:]=False
            ended=torch.where(done.cpu())[0]
            if len(ended):
                if self.observer:self.observer.close_slots(ended,False)
                if t+1<steps:
                    if ordinary:
                        started[ended]=True
                        if self.observer:self.observer.new_jobs(ended,torch.zeros(len(ended),dtype=torch.long))
                    else:
                        sources[ended]=0 if warmup else torch.multinomial(probs,len(ended),replacement=True,generator=generator)
                        sources[is_ref]=0;start_jobs(ended)
                elif ordinary:
                    # Vector wrapper has already reset; mark the next round's root.
                    started[ended]=True
                    if self.observer:self.observer.new_jobs(ended,torch.zeros(len(ended),dtype=torch.long))
        self.root_started=started.clone()
        b={k:torch.stack(v) for k,v in buffers.items()}
        fragments=[]
        low,high=low.cpu().reshape(-1,self.adapter.action_dim)[0],high.cpu().reshape(-1,self.adapter.action_dim)[0]
        for i in range(n):
            boundaries=[0]+[t+1 for t in range(steps-1) if b['term'][t,i] or b['trunc'][t,i]]+[steps]
            for left,right in zip(boundaries,boundaries[1:]):
                sl=slice(left,right);count=right-left;src=int(b['sources'][left,i])
                z=self.normalizer.normalize(b['next'][sl,i].clone())
                an=2*(b['applied'][sl,i]-low)/(high-low).clamp_min(1e-8)-1
                features=torch.cat([z,self.action_weight*an],-1)
                padded=torch.zeros(self.horizon,features.shape[-1]);mask=torch.zeros(self.horizon,dtype=torch.bool)
                padded[:count]=features;mask[:count]=True
                category='REFERENCE' if is_ref[i] else 'ROOT_ACQUISITION' if src==0 else 'REGION_ACQUISITION'
                f=RolloutFragment(src,version,self.fragment_id,b['obs'][sl,i].clone(),z,b['actions'][sl,i].clone(),b['rewards'][sl,i].clone(),
                    b['logprobs'][sl,i].clone(),b['values'][sl,i].clone(),b['term'][sl,i].clone(),b['trunc'][sl,i].clone(),mask,b['next_values'][right-1,i].clone(),
                    None,padded,b['next'][sl,i].clone(),b['next_values'][sl,i].clone(),chain_region_id=b['regions'][sl,i].clone(),
                    category=category,acquisition_round=version,root_started=src==0 and bool(b['starts'][left,i]),slot_id=i)
                self.fragment_id+=1;fragments.append(f)
        return fragments
