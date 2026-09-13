"""Fixed-window acquisition and fragment-local GAE for the local CPU runner.

All actual simulator transitions count, including references. This collector
uses one slot because PhysX GPU is unavailable on the target WSL machine.
"""
from dataclasses import dataclass
import torch

@dataclass
class AcquisitionJob:
    region_id: int
    snapshot: object = None
    max_steps: int = 32

@dataclass
class RolloutFragment:
    source_region_id: int
    policy_version: int
    fragment_id: int
    states: torch.Tensor
    state_features: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    logprobs: torch.Tensor
    values: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    valid_mask: torch.Tensor
    bootstrap_value: torch.Tensor
    entry_snapshot_id: int | None
    traj_features: torch.Tensor
    next_states: torch.Tensor
    next_values: torch.Tensor
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None
    chain_region_id: torch.Tensor | None = None
    category: str = 'ROOT_ACQUISITION'
    acquisition_round: int = 0
    root_started: bool = False
    slot_id: int = 0

    def gae(self,gamma,gae_lambda):
        adv = torch.zeros_like(self.rewards)
        carry = torch.zeros(())
        for t in reversed(range(len(adv))):
            delta = self.rewards[t]+gamma*self.next_values[t]*(~self.terminated[t])-self.values[t]
            carry = delta+gamma*gae_lambda*(~(self.terminated[t]|self.truncated[t]))*carry
            adv[t] = carry
        self.advantages,self.returns = adv,adv+self.values
        return adv,self.returns

    def batch(self):
        return dict(obs=self.states,actions=self.actions,logprobs=self.logprobs,
                    values=self.values,advantages=self.advantages,returns=self.returns)

class AcquisitionScheduler:
    def __init__(self,archive,horizon=32):
        self.archive,self.horizon = archive,horizon

    def make_jobs(self,allocation,num_envs=1):
        jobs = []
        for rid,n in allocation.items():
            for _ in range(n):
                snap = None if rid==0 else self.archive.sample_snapshot(self.archive.regions[rid])
                jobs.append(AcquisitionJob(rid,snap,self.horizon))
        return jobs

class FragmentCollector:
    def __init__(self,adapter,agent,normalizer,horizon=32,action_weight=1.,observer=None):
        if adapter.num_envs!=1:
            raise ValueError('Local fragment collector currently requires num_envs=1')
        self.adapter,self.agent,self.normalizer = adapter,agent,normalizer
        self.horizon,self.action_weight,self.observer = horizon,action_weight,observer
        self.obs = None
        self.fragment_id = 0
        self.counters = dict(ROOT_ACQUISITION=0,REGION_ACQUISITION=0,REFERENCE=0)

    @property
    def total_steps(self):
        return sum(self.counters.values())

    def collect(self,job,version,category=None,reset=True):
        category = category or ('ROOT_ACQUISITION' if job.region_id==0 else 'REGION_ACQUISITION')
        root_started = reset and job.region_id==0
        if reset:
            if job.region_id==0:
                self.obs,_ = self.adapter.reset()
            else:
                self.obs = self.adapter.restore_state(torch.tensor([0]),[job.snapshot.env_state])
                error = float((self.obs[0].cpu()-job.snapshot.obs).abs().max())
                if error>1e-4:
                    raise RuntimeError(f'Snapshot restore observation mismatch {error}')
            if self.observer:
                self.observer.new_fragment(job.region_id,root_started)
        elif self.obs is None:
            raise ValueError('Continuing before initial reset')
        rows=[]
        for t in range(job.max_steps):
            with torch.no_grad():
                dist = self.agent.get_distribution(self.obs)
                action,lp,_,value = self.agent.get_action_and_value(self.obs)
            rid = self.observer.observe(self.obs[0],dist.mean[0],self.agent.logstd[0],self.total_steps) if self.observer else 0
            applied = action.clamp(self.adapter.action_low(),self.adapter.action_high())
            nxt,reward,term,trunc,info = self.adapter.step(applied)
            self.counters[category] += 1
            done = bool(term[0]|trunc[0])
            actual_next = info['final_observation'] if done and 'final_observation' in info else nxt
            with torch.no_grad():
                nv = self.agent.get_value(actual_next)[0].cpu()
            rows.append((self.obs[0].cpu(),action[0].cpu(),reward[0].cpu(),lp[0].cpu(),value[0].cpu(),
                         term[0].cpu(),trunc[0].cpu(),actual_next[0].cpu(),nv,rid,applied[0].cpu()))
            self.obs = nxt
            if done:
                if self.observer:
                    self.observer.end_episode()
                break
        obs,a,r,lp,v,term,trunc,nxt,nv,rid,applied = zip(*rows)
        obs,a,r,lp,v,term,trunc,nxt,nv,applied = map(torch.stack,(obs,a,r,lp,v,term,trunc,nxt,nv,applied))
        z = self.normalizer.normalize(nxt)
        # Executed actions are normalized by the simulator's action-space scale.
        low,high = self.adapter.action_low().cpu().reshape(-1),self.adapter.action_high().cpu().reshape(-1)
        an = 2*(applied-low)/(high-low).clamp_min(1e-8)-1
        f = torch.cat([z,self.action_weight*an],-1)
        padded = torch.zeros(self.horizon,f.shape[-1]); valid = torch.zeros(self.horizon,dtype=torch.bool)
        n = min(len(f),self.horizon); padded[:n] = f[:n]; valid[:n] = True
        fragment = RolloutFragment(job.region_id,version,self.fragment_id,obs,z,a,r,lp,v,term,trunc,
            valid,nv[-1],None if job.snapshot is None else job.snapshot.timestep,padded,nxt,nv,
            chain_region_id=torch.tensor(rid),category=category,acquisition_round=version,root_started=root_started)
        self.fragment_id += 1
        return fragment


def concatenate_batches(fragments):
    batches = [f.batch() for f in fragments]
    return {k:torch.cat([b[k] for b in batches]) for k in batches[0]}
