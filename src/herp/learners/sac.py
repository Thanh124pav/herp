"""ManiSkill SAC learner: native objective, uniform replay, actor-only relevance."""
from collections import Counter
import copy
import torch
from torch import nn
from torch.nn import functional as F
from .sac_components import Actor, SoftQNetwork, ReplayBuffer, ReplayBufferSample


class RegionReplay(ReplayBuffer):
    """Region labels observe uniform sampling without affecting its probabilities."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.regions=torch.zeros((self.per_env_buffer_size,self.num_envs),dtype=torch.long)
        self.acquired=Counter(); self.sampled=Counter()

    def add(self, obs, next_obs, action, reward, done, region=0):
        labels=torch.as_tensor(region).cpu().expand(self.num_envs)
        self.regions[self.pos]=labels
        self.acquired.update(labels.tolist())
        super().add(obs,next_obs,action,reward,done)

    def sample(self,batch_size):
        count=self.per_env_buffer_size if self.full else self.pos
        if count==0:raise ValueError('Cannot sample empty replay')
        batch_inds=torch.randint(0,count,size=(batch_size,))
        env_inds=torch.randint(0,self.num_envs,size=(batch_size,))
        self.sampled.update(self.regions[batch_inds,env_inds].tolist())
        return ReplayBufferSample(**{key:getattr(self,key)[batch_inds,env_inds].to(self.sample_device)
                  for key in ('obs','next_obs','actions','rewards','dones')})

    def occupancy(self):
        count=self.per_env_buffer_size if self.full else self.pos
        return dict(Counter(self.regions[:count].flatten().tolist()))


class SACLearner:
    def __init__(self, env, args, device='cpu'):
        self.args=args;self.device=torch.device(device)
        self.actor=Actor(env).to(self.device)
        self.qf1=SoftQNetwork(env).to(self.device);self.qf2=SoftQNetwork(env).to(self.device)
        self.qf1_target=copy.deepcopy(self.qf1);self.qf2_target=copy.deepcopy(self.qf2)
        self.q_optimizer=torch.optim.Adam(list(self.qf1.parameters())+list(self.qf2.parameters()),lr=args.q_lr)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=args.policy_lr)
        self.log_alpha=torch.zeros(1,requires_grad=True,device=self.device)
        self.a_optimizer=torch.optim.Adam([self.log_alpha],lr=args.q_lr)
        self.target_entropy=-float(env.single_action_space.shape[0])
        self.alpha=self.log_alpha.exp().item() if args.autotune else args.alpha
        self.replay=RegionReplay(env,args.num_envs,args.buffer_size,torch.device(args.buffer_device),self.device)
        self.global_update=0

    @torch.no_grad()
    def act(self,obs,deterministic=False):
        return self.actor.get_eval_action(obs) if deterministic else self.actor.get_action(obs)[0]

    def observe(self,transition):
        self.replay.add(**transition)

    def policy_parameters(self):
        return list(self.actor.parameters())

    def gradient_signature(self,batch):
        obs=batch['obs'].to(self.device)
        action,log_pi,_=self.actor.get_action(obs)
        loss=(self.alpha*log_pi-torch.minimum(self.qf1(obs,action),self.qf2(obs,action))).mean()
        grads=torch.autograd.grad(loss,self.policy_parameters())
        return torch.cat([g.detach().flatten() for g in grads])

    def update(self):
        a=self.args;self.global_update+=1
        data=self.replay.sample(a.batch_size)
        with torch.no_grad():
            action,log_pi,_=self.actor.get_action(data.next_obs)
            q=torch.minimum(self.qf1_target(data.next_obs,action),self.qf2_target(data.next_obs,action))-self.alpha*log_pi
            target=data.rewards.flatten()+(1-data.dones.flatten())*a.gamma*q.view(-1)
        q1=self.qf1(data.obs,data.actions).view(-1);q2=self.qf2(data.obs,data.actions).view(-1)
        loss1=F.mse_loss(q1,target);loss2=F.mse_loss(q2,target)
        self.q_optimizer.zero_grad();(loss1+loss2).backward();self.q_optimizer.step()
        metrics=dict(qf1_loss=loss1.item(),qf2_loss=loss2.item(),qf1_values=q1.mean().item(),qf2_values=q2.mean().item())
        if self.global_update%a.policy_frequency==0:
            pi,log_pi,_=self.actor.get_action(data.obs)
            actor_loss=(self.alpha*log_pi-torch.minimum(self.qf1(data.obs,pi),self.qf2(data.obs,pi))).mean()
            self.actor_optimizer.zero_grad();actor_loss.backward();self.actor_optimizer.step()
            metrics['actor_loss']=actor_loss.item()
            if a.autotune:
                with torch.no_grad():_,log_pi,_=self.actor.get_action(data.obs)
                alpha_loss=(-self.log_alpha.exp()*(log_pi+self.target_entropy)).mean()
                self.a_optimizer.zero_grad();alpha_loss.backward();self.a_optimizer.step()
                self.alpha=self.log_alpha.exp().item();metrics['alpha_loss']=alpha_loss.item()
        if self.global_update%a.target_network_frequency==0:
            with torch.no_grad():
                for source,target in ((self.qf1,self.qf1_target),(self.qf2,self.qf2_target)):
                    for param,target_param in zip(source.parameters(),target.parameters()):
                        target_param.copy_(a.tau*param+(1-a.tau)*target_param)
        metrics['alpha']=self.alpha
        return metrics

    def save(self,path):
        state={k:getattr(self,k).state_dict() for k in ('actor','qf1','qf2','qf1_target','qf2_target','q_optimizer','actor_optimizer','a_optimizer')}
        state.update(log_alpha=self.log_alpha.detach(),alpha=self.alpha,global_update=self.global_update,
                     replay=self.replay,torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        torch.save(state,path)

    def load(self,path):
        # RegionReplay is the only non-tensor object in learner checkpoints.
        # Keep PyTorch's restricted weights-only unpickler enabled and
        # allowlist only this repository-owned container.
        with torch.serialization.safe_globals([RegionReplay]):
            state=torch.load(path,weights_only=True,map_location=self.device)
        for k in ('actor','qf1','qf2','qf1_target','qf2_target','q_optimizer','actor_optimizer','a_optimizer'):
            getattr(self,k).load_state_dict(state[k])
        with torch.no_grad():self.log_alpha.copy_(state['log_alpha'])
        self.alpha=state['alpha'];self.global_update=state['global_update'];self.replay=state['replay']
        self.replay.storage_device=self.replay.obs.device;self.replay.sample_device=self.device
        self.replay.regions=self.replay.regions.cpu()
        torch.set_rng_state(state['torch_rng'].cpu())
        if state['cuda_rng'] is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
