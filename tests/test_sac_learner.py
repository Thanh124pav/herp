import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import gymnasium as gym
import numpy as np
import torch
from torch.nn import functional as F
from herp.learners.sac import SACLearner, RegionReplay
from herp.learners.sac_components import ReplayBuffer


def setup():
    torch.set_num_threads(1)
    env=SimpleNamespace(single_observation_space=gym.spaces.Box(-1,1,(3,),dtype=np.float32),
                        single_action_space=gym.spaces.Box(-1,1,(2,),dtype=np.float32))
    args=SimpleNamespace(q_lr=3e-4,policy_lr=3e-4,num_envs=1,buffer_size=32,buffer_device='cpu',
        autotune=True,alpha=.2,batch_size=8,gamma=.8,tau=.01,policy_frequency=1,target_network_frequency=1,
        grad_steps_per_iteration=1)
    learner=SACLearner(env,args)
    for i in range(12):
        learner.observe(dict(obs=torch.randn(1,3),next_obs=torch.randn(1,3),action=torch.rand(1,2),
                             reward=torch.rand(1),done=torch.zeros(1),region=i%3))
    return learner,env,args


def test_uniform_replay_parity_and_overwrite():
    learner,env,args=setup();rb=learner.replay
    plain=ReplayBuffer(env,1,32,torch.device('cpu'),torch.device('cpu'))
    for i in range(rb.pos):plain.add(rb.obs[i],rb.next_obs[i],rb.actions[i],rb.rewards[i],rb.dones[i])
    rng=torch.get_rng_state();a=rb.sample(100);torch.set_rng_state(rng);b=plain.sample(100)
    assert all(torch.equal(getattr(a,k),getattr(b,k)) for k in vars(a))
    assert sum(rb.sampled.values())==100
    for i in range(40):rb.add(torch.zeros(1,3),torch.zeros(1,3),torch.zeros(1,2),torch.zeros(1),torch.zeros(1),7)
    assert rb.occupancy()=={7:32}


def test_signature_is_actor_only_and_does_not_accumulate_gradients():
    learner,_,_=setup()
    g=learner.gradient_signature({'obs':torch.randn(8,3)})
    assert len(g)==sum(p.numel() for p in learner.actor.parameters())
    assert torch.isfinite(g).all() and g.norm()>0
    assert all(p.grad is None for p in learner.actor.parameters())
    assert all(p.grad is None for p in learner.qf1.parameters())


def test_update_matches_actual_upstream_loop():
    learner,_,args=setup();other=copy.deepcopy(learner)
    path=Path(__file__).resolve().parents[1]/'third_party/ManiSkill/examples/baselines/sac/sac.py'
    if not path.exists():
        import pytest;pytest.skip('Missing locked upstream')
    tree=ast.parse(path.read_text())
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='local_update')
    module=ast.Module(body=[loop],type_ignores=[])
    scope={k:getattr(other,k) for k in ('actor','qf1','qf2','qf1_target','qf2_target','q_optimizer','actor_optimizer','a_optimizer','log_alpha','alpha','target_entropy')}
    scope.update(args=args,torch=torch,F=F,rb=other.replay,global_update=0)
    rng=torch.get_rng_state();learner.update();torch.set_rng_state(rng)
    exec(compile(module,str(path),'exec'),scope)
    for name in ('actor','qf1','qf2','qf1_target','qf2_target'):
        assert all(torch.equal(a,b) for a,b in zip(getattr(learner,name).parameters(),getattr(other,name).parameters())),name
    assert learner.alpha==scope['alpha']


def test_checkpoint_continues_same_update(tmp_path):
    learner,_,_=setup();learner.update();path=tmp_path/'sac.pt';learner.save(path)
    expected=learner.update();weights=copy.deepcopy(learner.actor.state_dict())
    learner.load(path);actual=learner.update()
    assert expected==actual
    assert all(torch.equal(v,learner.actor.state_dict()[k]) for k,v in weights.items())
